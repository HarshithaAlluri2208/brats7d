export type User = {
  id: string;
  name: string;
  email: string;
};

export type AuthResult = {
  token: string;
  user: User;
};

const STORAGE_KEY = "neurovision_auth";

export function getSession(): AuthResult | null {
  if (typeof window === "undefined") return null;
  const raw = localStorage.getItem(STORAGE_KEY);
  if (!raw) return null;
  try { return JSON.parse(raw) as AuthResult; } catch { return null; }
}

export function setSession(session: AuthResult): void {
  if (typeof window === "undefined") return;
  localStorage.setItem(STORAGE_KEY, JSON.stringify(session));
}

export function clearSession(): void {
  if (typeof window === "undefined") return;
  localStorage.removeItem(STORAGE_KEY);
}

function fakeToken(): string {
  return `mock.${Math.random().toString(36).slice(2)}.${Date.now()}`;
}

export async function signup(name: string, email: string, password: string): Promise<AuthResult> {
  await new Promise((r) => setTimeout(r, 600));
  const session: AuthResult = {
    token: fakeToken(),
    user: { id: crypto.randomUUID?.() || String(Date.now()), name, email },
  };
  setSession(session);
  return session;
}

export async function login(email: string, password: string): Promise<AuthResult> {
  await new Promise((r) => setTimeout(r, 500));
  const session: AuthResult = {
    token: fakeToken(),
    user: { id: crypto.randomUUID?.() || String(Date.now()), name: email.split("@")[0], email },
  };
  setSession(session);
  return session;
}


