"use client"

import { signOut } from "next-auth/react"
import { useRouter } from "next/navigation"
import { FiLogOut } from "react-icons/fi"

export default function SignOutButton() {
    const router = useRouter()

    const handleSignOut = async () => {
        await signOut({
            redirect: false
        })
        router.push("/login")
    }

    return <button onClick={handleSignOut} aria-label="Sign out" className="flex h-9 items-center gap-2 rounded bg-white px-4 text-sm font-medium text-[#003d7a] transition-colors hover:bg-[#f0f0f0]">
        <FiLogOut className="h-4 w-4" />
        <span className="hidden sm:inline">Logout</span>
    </button>
}