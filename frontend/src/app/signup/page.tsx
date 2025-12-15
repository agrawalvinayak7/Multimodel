"use client";

import { zodResolver } from "@hookform/resolvers/zod";
import { signupSchema, type SignupSchema } from "~/schemas/auth";
import { useForm } from "react-hook-form";
import { useState } from "react";
import Link from "next/link";
import { registerUser } from "~/app/actions/auth";
import { signIn } from "next-auth/react";
import { useRouter } from "next/navigation";

export default function SignupPage() {
    const router = useRouter();
    const [error, setError] = useState<string>("");
    const [loading, setLoading] = useState<boolean>(false);
    const form = useForm<SignupSchema>({
        resolver: zodResolver(signupSchema),
        defaultValues: {
            name: "",
            email: "",
            password: "",
            confirmPassword: "",
        },
    });

    async function onSubmit(data: SignupSchema) {
        try {
            setLoading(true);
            const result = await registerUser(data);

            if (result.error) {
                setError(result.error);
                return;
            }

            // sign in after registration
            const signInResult = await signIn("credentials", {
                email: data.email,
                password: data.password,
                redirect: false,
            });
            if (!signInResult?.error) {
                router.push("/");
            } else {
                setError("Failed to sign in");
            }
        } catch {
            setError("Something went wrong");
        } finally {
            setLoading(false);
        }
    }

    return (
        <div
            className="min-h-screen bg-cover bg-center bg-fixed flex items-center justify-end px-8"
            style={{ backgroundImage: "url('/background.jpg')" }}
        >
            {/* Right Side Card */}
            <div className="animate-fade-in w-full max-w-md mr-8 lg:mr-16">
                <div
                    className="rounded-lg p-10 border border-[#999999]"
                    style={{ backgroundColor: '#d4d5cf' }}
                >
                    {/* Header */}
                    <div className="mb-8">
                        <h2 className="text-[1.75rem] font-bold text-[#1d1d1d] mb-2">
                            Create Account
                        </h2>
                        <p className="text-sm text-[#666666]">
                            Sign up to get started with sentiment analysis
                        </p>
                    </div>

                    <form className="space-y-4" onSubmit={form.handleSubmit(onSubmit)}>
                        {error && (
                            <div className="rounded border border-[#d32f2f] bg-[#ffebee] p-3 text-center text-sm text-[#d32f2f]">
                                {error}
                            </div>
                        )}

                        <div className="space-y-4">
                            <div>
                                <label className="block text-xs font-semibold text-[#1d1d1d] mb-1">
                                    Full Name
                                </label>
                                <input
                                    {...form.register("name")}
                                    type="text"
                                    placeholder="Vinayak Agrawal"
                                    className="w-full rounded bg-white border border-[#999999] px-3 py-2 text-sm text-[#1d1d1d] placeholder-[#999999] outline-none transition-all duration-200 focus:border-[#5b7fc4] focus:border-2"
                                />
                                {form.formState.errors.name && (
                                    <p className="mt-1 text-xs text-[#d32f2f]">
                                        {form.formState.errors.name.message}
                                    </p>
                                )}
                            </div>

                            <div>
                                <label className="block text-xs font-semibold text-[#1d1d1d] mb-1">
                                    Email address
                                </label>
                                <input
                                    {...form.register("email")}
                                    type="email"
                                    placeholder="example@gmail.com"
                                    className="w-full rounded bg-white border border-[#999999] px-3 py-2 text-sm text-[#1d1d1d] placeholder-[#999999] outline-none transition-all duration-200 focus:border-[#5b7fc4] focus:border-2"
                                />
                                {form.formState.errors.email && (
                                    <p className="mt-1 text-xs text-[#d32f2f]">
                                        {form.formState.errors.email.message}
                                    </p>
                                )}
                            </div>

                            <div>
                                <label className="block text-xs font-semibold text-[#1d1d1d] mb-1">
                                    Password
                                </label>
                                <input
                                    {...form.register("password")}
                                    type="password"
                                    placeholder="••••••••"
                                    className="w-full rounded bg-white border border-[#999999] px-3 py-2 text-sm text-[#1d1d1d] placeholder-[#999999] outline-none transition-all duration-200 focus:border-[#5b7fc4] focus:border-2"
                                />
                                {form.formState.errors.password && (
                                    <p className="mt-1 text-xs text-[#d32f2f]">
                                        {form.formState.errors.password.message}
                                    </p>
                                )}
                            </div>

                            <div>
                                <label className="block text-xs font-semibold text-[#1d1d1d] mb-1">
                                    Confirm Password
                                </label>
                                <input
                                    {...form.register("confirmPassword")}
                                    type="password"
                                    placeholder="••••••••"
                                    className="w-full rounded bg-white border border-[#999999] px-3 py-2 text-sm text-[#1d1d1d] placeholder-[#999999] outline-none transition-all duration-200 focus:border-[#5b7fc4] focus:border-2"
                                />
                                {form.formState.errors.confirmPassword && (
                                    <p className="mt-1 text-xs text-[#d32f2f]">
                                        {form.formState.errors.confirmPassword.message}
                                    </p>
                                )}
                            </div>
                        </div>

                        <button
                            type="submit"
                            disabled={loading}
                            className="w-full rounded-md bg-[#5b7fc4] px-6 py-2.5 text-sm font-semibold text-white transition-colors duration-200 hover:bg-[#4a6ba3] disabled:opacity-60 disabled:cursor-not-allowed"
                        >
                            {loading ? "Creating account..." : "Create account"}
                        </button>

                        <p className="text-center text-sm text-[#666666]">
                            Already have an account?{" "}
                            <Link
                                href="/login"
                                className="font-medium text-[#5b7fc4] hover:underline"
                            >
                                Sign in
                            </Link>
                        </p>
                    </form>
                </div>
            </div>
        </div>
    );
}