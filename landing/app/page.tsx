import { Nav } from "@/components/Nav";
import { Hero } from "@/components/Hero";
import { Features } from "@/components/Features";
import { CodeExample } from "@/components/CodeExample";
import { Waitlist } from "@/components/Waitlist";
import { Footer } from "@/components/Footer";

export default function Home() {
  return (
    <>
      <Nav />
      <main>
        <Hero />
        <CodeExample />
        <Features />
        <Waitlist />
      </main>
      <Footer />
    </>
  );
}
