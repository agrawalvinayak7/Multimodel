"use server";

import { signupSchema, type SignupSchema } from "~/schemas/auth";
import { db } from "~/server/db";
import bcrypt from "bcryptjs";

export async function registerUser(data: SignupSchema) {
    try {
    //server side validation
    const result = signupSchema.safeParse(data);
    if(!result.success){
        return {error: "Invalid data"}
        }
    
        const {name, email, password} = data;

        //check if user already exists

        const existingUser = await db.user.findUnique({
            where: {email}
        })
        if(existingUser){
            return {error: "User already exists"}
        }

        //hash password
        const hashedPassword = await bcrypt.hash(password, 12)
        //create user
        await db.user.create({
            data: {
                name,
                email,
                password: hashedPassword
            }
        })


        return {success: true}
    } catch(error){
        return {error: "Something went wrong"}
    }
}
