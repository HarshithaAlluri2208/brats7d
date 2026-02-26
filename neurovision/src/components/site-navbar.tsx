"use client";

import Link from "next/link";
import { motion } from "framer-motion";
import { Button } from "@/components/ui/button";
import { LaunchButton } from "@/components/launch-button";
import { useEffect, useState } from "react";
import { getSession, clearSession } from "@/lib/auth";
import { useRouter } from "next/navigation";

export function SiteNavbar() {
  const router = useRouter();
  const [isAuthed, setIsAuthed] = useState(false);

  useEffect(() => {
    setIsAuthed(!!getSession());
  }, []);

  function signOut() {
    clearSession();
    setIsAuthed(false);
    router.push("/");
  }

  return (
    <header className="sticky top-0 z-40 w-full border-b bg-background/70 backdrop-blur supports-[backdrop-filter]:bg-background/60">
      <div className="container mx-auto flex h-14 items-center justify-between px-6">
        <motion.div initial={{ opacity: 0, y: -6 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.3 }}>
          <Link href="/" className="font-semibold tracking-tight">
            NeuroVision
          </Link>
        </motion.div>
        <nav className="flex items-center gap-3 text-sm">
          <Link href="/dashboard" className="text-muted-foreground hover:text-foreground transition-colors">
            Dashboard
          </Link>
          <Link href="/visualization" className="text-muted-foreground hover:text-foreground transition-colors">
            Visualization
          </Link>
          <Link href="/report" className="text-muted-foreground hover:text-foreground transition-colors">
            Report
          </Link>
          {!isAuthed ? (
            <>
              <Link href="/auth/login" className="text-muted-foreground hover:text-foreground transition-colors">
                Sign in
              </Link>
              <Link href="/auth/signup" className="text-muted-foreground hover:text-foreground transition-colors">
                Sign up
              </Link>
            </>
          ) : (
            <Button variant="ghost" size="sm" onClick={signOut} className="text-muted-foreground hover:text-foreground">
              Sign out
            </Button>
          )}
          <LaunchButton size="sm" className="ml-2">Launch</LaunchButton>
        </nav>
      </div>
    </header>
  );
}


