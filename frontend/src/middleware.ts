import NextAuth from "next-auth";
import { type NextAuthConfig } from "next-auth";

// Lightweight config for middleware (no Prisma adapter to reduce bundle size)
// Must match the main auth config's JWT settings to validate tokens correctly.
// We use process.env.AUTH_SECRET directly to avoid importing the `~/env` module,
// which might pull in other heavy dependencies.
const middlewareConfig = {
  pages: {
    signIn: "/login",
  },
  // Providers are technically required by NextAuthConfig, but we'll leave this empty
  // as they are not needed for JWT validation in the middleware itself.
  providers: [],
  secret: process.env.AUTH_SECRET, // IMPORTANT: Must match the AUTH_SECRET in your Vercel Environment Variables
  session: {
    strategy: "jwt" as const,
  },
  callbacks: {
    // This callback is crucial: it determines if the user is authenticated.
    // If 'auth' (the session token) exists, the user is authorized.
    authorized: ({ auth }) => !!auth,
    // These callbacks are for JWT handling and session creation.
    // They should mirror your main `authConfig` if you're using JWT strategy.
    jwt: ({ token }) => token,
    session: ({ session, token }) => ({
      ...session,
      user: {
        ...session.user,
        id: token.sub, // Ensure the user ID is correctly transferred
      },
    }),
  },
} satisfies NextAuthConfig;

const { auth } = NextAuth(middlewareConfig);

export default auth((req) => {
  // IMPORTANT: For debugging, you can add a check for AUTH_SECRET here locally.
  // In production, Vercel will ensure env vars are present during build.
  // if (!process.env.AUTH_SECRET) {
  //   console.error("AUTH_SECRET is missing in environment variables for middleware!");
  //   // Optionally, you might want to redirect to an error page or allow access
  //   // depending on your security policy when AUTH_SECRET is not configured.
  //   // For now, it will simply proceed without authentication which will lead to a redirect to /login anyway.
  // }

  const isAuthenticated = !!req.auth;

  // If the user is not authenticated and is trying to access a protected route,
  // redirect them to the login page.
  if (!isAuthenticated) {
    const newUrl = new URL("/login", req.nextUrl.origin);
    return Response.redirect(newUrl);
  }
});

export const config = {
  // The matcher configuration specifies which paths the middleware should run on.
  // We want to protect most routes but explicitly exclude authentication-related paths.
  matcher: [
    /*
     * Match all request paths EXCEPT for the ones starting with:
     * - /api/auth (NextAuth.js authentication routes for login/signup)
     * - /login (the login page itself)
     * - /signup (the signup page itself)
     * - /_next/static (Next.js static assets)
     * - /_next/image (Next.js image optimization routes)
     * - /favicon.ico (the favicon file)
     *
     * The regex `(?!...)` is a negative lookahead, meaning "don't match if it starts with any of these".
     * The `.*` at the end means "match anything that follows".
     */
    "/((?!api/auth|login|signup|_next/static|_next/image|favicon.ico).*)",
  ],
};