"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { motion } from "framer-motion";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { getSession, clearSession } from "@/lib/auth";

export default function ProfilePage() {
  const router = useRouter();
  const [name, setName] = useState("");
  const [age, setAge] = useState("");
  const [mrn, setMrn] = useState("");

  useEffect(() => {
    const session = getSession();
    if (!session) {
      router.replace("/auth/login");
    } else {
      setName(session.user.name || "");
    }
  }, [router]);

  function logout() {
    clearSession();
    router.push("/auth/login");
  }

  function onSubmit(e: React.FormEvent) {
    e.preventDefault();
    router.push("/dashboard");
  }

  return (
    <div className="container mx-auto px-6 py-12 max-w-xl">
      <motion.h1 initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} className="text-3xl font-semibold tracking-tight">
        Patient Metadata
      </motion.h1>
      <p className="text-muted-foreground mt-1">Complete patient details before proceeding.</p>
      <Card className="mt-6">
        <CardHeader>
          <CardTitle>Profile</CardTitle>
        </CardHeader>
        <CardContent>
          <form onSubmit={onSubmit} className="space-y-4">
            <div className="space-y-2">
              <label className="text-sm" htmlFor="name">Name</label>
              <Input id="name" value={name} onChange={(e) => setName(e.target.value)} />
            </div>
            <div className="space-y-2">
              <label className="text-sm" htmlFor="age">Age</label>
              <Input id="age" value={age} onChange={(e) => setAge(e.target.value)} />
            </div>
            <div className="space-y-2">
              <label className="text-sm" htmlFor="mrn">MRN</label>
              <Input id="mrn" value={mrn} onChange={(e) => setMrn(e.target.value)} />
            </div>
            <div className="flex items-center gap-3">
              <Button type="submit">Save & Continue</Button>
              <Button type="button" variant="secondary" onClick={logout}>Log out</Button>
            </div>
          </form>
        </CardContent>
      </Card>
    </div>
  );
}


