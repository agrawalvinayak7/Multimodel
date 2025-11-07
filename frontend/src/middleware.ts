import NextAuth from "next-auth";
import { type NextAuthConfig } from "next-auth";

// Lightweight config for middleware (no Prisma adapter to reduce bundle size)
// Must match the main auth config's JWT settings to validate tokens correctly
// Use process.env directly to avoid importing env module (which might pull in Prisma)
const middlewareConfig = {
  pages: {
    signIn: "/login",
  },
  providers: [], // Empty array - providers not needed for JWT validation in middleware
  secret: process.env.AUTH_SECRET, // Must match the main auth config secret
  session: {
    strategy: "jwt" as const,
  },
  callbacks: {
    authorized: ({ auth }) => !!auth,
    jwt: ({ token }) => token, // Pass through JWT token
    session: ({ session, token }) => ({
      ...session,
      user: {
        ...session.user,
        id: token.sub,
      },
    }),
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
  // Only protect the root path, exclude auth routes and login/signup pages
  matcher: [
    /*
     * Match all request paths except for the ones starting with:
     * - api/auth (NextAuth routes)
     * - login (login page)
     * - signup (signup page)
     * - _next/static (static files)
     * - _next/image (image optimization files)
     * - favicon.ico (favicon file)
     */
    "/((?!api/auth|login|signup|_next/static|_next/image|favicon.ico).*)",
  ],
};
