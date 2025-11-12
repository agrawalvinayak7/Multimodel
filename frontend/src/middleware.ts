import type { NextRequest } from "next/server";
import { NextResponse } from "next/server";
import { getToken } from "next-auth/jwt";

const authSecret = process.env.AUTH_SECRET;

if (!authSecret) {
    throw new Error("AUTH_SECRET is not set. Add it to your environment to run the middleware.");
}

export async function middleware(req: NextRequest) {
    const session = await getToken({ req, secret: authSecret });

    if (!session) {
        const loginUrl = new URL("/login", req.nextUrl.origin);
        loginUrl.searchParams.set("from", req.nextUrl.pathname + req.nextUrl.search);
        return NextResponse.redirect(loginUrl);
    }

    return NextResponse.next();
}

export const config = {
    matcher: ["/"],
};
