"use server";

import CodeExamples from "~/components/client/CodeExamples";
import CopyButton from "~/components/client/copy-button";
import { Inference } from "~/components/client/inference";
import SignOutButton from "~/components/client/Signout";
import { auth } from "~/server/auth";
import { db } from "~/server/db";
import LoginPage from "~/app/login/page";
import { Logo } from "~/components/ui/logo";

export default async function HomePage() {
  const session = await auth();
  if (!session?.user?.id) {
    return <LoginPage />;
  }
  const quota = await db.apiQuota.findUniqueOrThrow({
    where: {
      userId: session?.user.id,
    },
  });

  return (
    <div className="min-h-screen w-full bg-[#004d9c]">
      <nav className="sticky top-0 z-50 bg-[#003d7a] border-b border-[#0055b3]">
        <div className="mx-auto flex h-16 max-w-7xl items-center justify-between px-6">
          <div className="flex items-center gap-3">
            <Logo className="h-8 w-8" />
            <span className="text-xl font-semibold text-white">Sentiment Analysis</span>
          </div>

          <SignOutButton />
        </div>
      </nav>

      <main className="mx-auto flex max-w-7xl flex-col gap-6 p-6 md:flex-row md:p-8">
        {/* Left Column: Inference */}
        <div className="flex w-full flex-col gap-6 md:w-7/12 lg:w-3/5">
          <div className="rounded-lg bg-white border border-[#cccccc] p-6">
            <Inference quota={{ secretKey: quota.secretKey }} />
          </div>
        </div>

        {/* Right Column: API & Info */}
        <div className="flex w-full flex-col gap-6 md:w-5/12 lg:w-2/5">

          {/* API Key Card */}
          <div className="rounded-lg bg-white border border-[#cccccc] p-6">
            <div className="border-b border-[#e0e0e0] pb-4 mb-4">
              <h2 className="text-base font-semibold text-[#1d1d1d]">Secret key</h2>
            </div>

            <div className="mb-4">
              <p className="text-sm text-[#666666] leading-relaxed">
                Use this key to authenticate your requests. Keep it safe!
              </p>
            </div>

            <div className="flex items-center justify-between rounded bg-[#f5f5f5] border border-[#cccccc] p-3">
              <span className="font-mono text-sm text-[#1d1d1d] truncate max-w-[200px]">
                {quota.secretKey}
              </span>
              <CopyButton text={quota.secretKey} />
            </div>
          </div>

          {/* Quota Card */}
          <div className="rounded-lg bg-white border border-[#cccccc] p-6">
            <div className="flex items-center justify-between border-b border-[#e0e0e0] pb-4 mb-4">
              <h2 className="text-base font-semibold text-[#1d1d1d]">Monthly quota</h2>
              <span className="text-sm font-medium text-[#666666]">
                {quota.requestsUsed} / {quota.maxRequests}
              </span>
            </div>

            <div className="h-2 w-full overflow-hidden rounded-full bg-[#e0e0e0]">
              <div
                style={{
                  width: (quota.requestsUsed / quota.maxRequests) * 100 + "%",
                }}
                className="h-full rounded-full bg-[#5b7fc4] transition-all duration-500"
              ></div>
            </div>
          </div>

          {/* Code Examples */}
          <CodeExamples />
        </div>
      </main>
    </div>
  );
}