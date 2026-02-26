"use client";

import { Button } from "@/components/ui/button";
import { getSession } from "@/lib/auth";
import { useRouter } from "next/navigation";

type Props = {
  size?: "default" | "sm" | "lg" | "icon";
  className?: string;
  children?: React.ReactNode;
};

export function LaunchButton({ size = "default", className, children }: Props) {
  const router = useRouter();

  function onClick() {
    const session = getSession();
    router.push(session ? "/dashboard" : "/auth/login");
  }

  return (
    <Button size={size} className={className} onClick={onClick}>
      {children ?? "Launch"}
    </Button>
  );
}


