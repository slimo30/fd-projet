import { type NextRequest, NextResponse } from "next/server"

export async function POST(request: NextRequest) {
  try {
    const formData = await request.formData()

    // Forward the request to the Flask API
    const response = await fetch("http://127.0.0.1:5001/upload-data", {
      method: "POST",
      body: formData,
    })

    if (!response.ok) {
      throw new Error(`Flask API error: ${response.statusText}`)
    }

    const data = await response.json()
    return NextResponse.json(data)
  } catch (error) {
    console.error("Error uploading file:", error)
    return NextResponse.json({ error: "Failed to upload file" }, { status: 500 })
  }
}
