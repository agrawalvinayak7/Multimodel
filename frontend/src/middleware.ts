import NextAuth from "next-auth";
import { type NextAuthConfig } from "next-auth";

// Lightweight config for middleware (no Prisma adapter to reduce bundle size)
// Providers are required but won't be used in middleware
const middlewareConfig = {
  pages: {
    signIn: "/login",
  },
  providers: [], // Empty array - providers not needed for JWT validation in middleware
  session: {
    strategy: "jwt" as const,
  },
  callbacks: {
    authorized: ({ auth }) => !!auth,
  },
} satisfies NextAuthConfig;

const { auth } = NextAuth(middlewareConfig);

export default auth((req) => {
  const isAuthenticated = !!req.auth;

  if (!isAuthenticated) {
    const newUrl = new URL("/login", req.nextUrl.origin);
    return Response.redirect(newUrl);
  }
});

export const config = {
  matcher: ["/"],
};
