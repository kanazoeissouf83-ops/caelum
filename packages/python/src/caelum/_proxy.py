"""Composition-based proxies that make CAELUM wrappers drop-in replacements.

WHY composition and not inheritance: subclassing ``openai.OpenAI`` would tie CAELUM to
private internals that change between SDK releases (constructors, cached properties,
``copy``). Wrapping an untouched SDK client and forwarding everything we do not
intercept keeps CAELUM robust to upstream changes: new SDK methods work on day one.

For static type checkers, each wrapper *declares* the SDK client as its base class
(under ``TYPE_CHECKING``) so users keep full autocompletion and typing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from caelum._engine import Endpoint, Engine

_INTERNAL_PREFIX = "_caelum_"


class ResourceProxy:
    """Forward unknown attribute access to a wrapped SDK resource (e.g. ``chat``)."""

    def __init__(self, target: Any) -> None:
        self._caelum_target = target

    def __getattr__(self, name: str) -> Any:
        if name.startswith(_INTERNAL_PREFIX):
            raise AttributeError(name)
        return getattr(self._caelum_target, name)

    def __dir__(self) -> list[str]:
        return sorted(set(dir(self._caelum_target)) | set(super().__dir__()))

    def __repr__(self) -> str:
        return f"<caelum proxy for {self._caelum_target!r}>"


class InterceptedResource(ResourceProxy):
    """SDK resource whose ``create`` method goes through the CAELUM engine.

    Every other attribute (``stream``, ``with_raw_response``, ``batches``...) is
    forwarded untouched to the SDK resource.
    """

    def __init__(self, target: Any, engine: Engine, endpoint: Endpoint) -> None:
        super().__init__(target)
        self._caelum_engine = engine
        self._caelum_endpoint = endpoint

    def create(self, *args: Any, **kwargs: Any) -> Any:
        """The SDK ``create`` method, routed through the CAELUM pipeline."""
        target = self._caelum_target
        if args:  # SDKs only accept keyword arguments: let them raise as usual.
            return target.create(*args, **kwargs)
        return self._caelum_engine.execute(
            self._caelum_endpoint, kwargs, lambda request: target.create(**request)
        )


class AsyncInterceptedResource(ResourceProxy):
    """Asynchronous counterpart of :class:`InterceptedResource`."""

    def __init__(self, target: Any, engine: Engine, endpoint: Endpoint) -> None:
        super().__init__(target)
        self._caelum_engine = engine
        self._caelum_endpoint = endpoint

    async def create(self, *args: Any, **kwargs: Any) -> Any:
        """The SDK ``create`` method, routed through the CAELUM pipeline."""
        target = self._caelum_target
        if args:
            return await target.create(*args, **kwargs)
        return await self._caelum_engine.aexecute(
            self._caelum_endpoint, kwargs, lambda request: target.create(**request)
        )


class ClientProxy:
    """Base behaviour shared by every CAELUM client wrapper.

    Attributes starting with ``_caelum_`` live on the wrapper. Every other attribute is
    read from (and written to) the wrapped SDK client, except the intercepted resources
    registered in ``_caelum_resources`` (``chat``, ``embeddings``, ``messages``...).
    """

    _caelum_client: Any
    _caelum_resources: dict[str, Any]

    def __getattr__(self, name: str) -> Any:
        # Only called when normal lookup fails.
        if name.startswith(_INTERNAL_PREFIX):
            raise AttributeError(name)
        resources: dict[str, Any] | None = self.__dict__.get("_caelum_resources")
        if resources is not None and name in resources:
            return resources[name]
        return getattr(self._caelum_client, name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name.startswith(_INTERNAL_PREFIX):
            object.__setattr__(self, name, value)
        else:
            # WHY: `client.api_key = ...` must update the real client, like the SDK.
            setattr(self._caelum_client, name, value)

    def __delattr__(self, name: str) -> None:
        if name.startswith(_INTERNAL_PREFIX):
            object.__delattr__(self, name)
        else:
            delattr(self._caelum_client, name)

    def __dir__(self) -> list[str]:
        return sorted(
            set(dir(self._caelum_client)) | set(self._caelum_resources) | set(super().__dir__())
        )

    def __repr__(self) -> str:
        return f"<caelum.{type(self).__name__} wrapping {self._caelum_client!r}>"

    def _caelum_rewrap(self, client: Any) -> Any:
        """Wrap another SDK client while sharing this wrapper's CAELUM state."""
        raise NotImplementedError  # pragma: no cover - implemented by wrappers

    def copy(self, *args: Any, **kwargs: Any) -> Any:
        """Same as the SDK ``copy``, but the copy stays wrapped by CAELUM."""
        return self._caelum_rewrap(self._caelum_client.copy(*args, **kwargs))

    def with_options(self, *args: Any, **kwargs: Any) -> Any:
        """Same as the SDK ``with_options``, but the result stays wrapped by CAELUM."""
        return self._caelum_rewrap(self._caelum_client.with_options(*args, **kwargs))


def _pool(wrapper: object) -> Any:
    """Provider pool of a wrapper's CAELUM state, if any."""
    state = wrapper.__dict__.get("_caelum_state")
    return getattr(state, "providers", None)


class SyncClientProxy(ClientProxy):
    """Client proxy with synchronous context-manager support."""

    def close(self) -> None:
        """Close the underlying SDK client (and the fallback/routing clients CAELUM made)."""
        self._caelum_client.close()
        pool = _pool(self)
        if pool is not None:
            pool.close()

    def __enter__(self) -> Any:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


class AsyncClientProxy(ClientProxy):
    """Client proxy with asynchronous context-manager support."""

    async def close(self) -> None:
        """Close the underlying SDK client (and the fallback/routing clients CAELUM made)."""
        await self._caelum_client.close()
        pool = _pool(self)
        if pool is not None:
            await pool.aclose()

    async def __aenter__(self) -> Any:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.close()
