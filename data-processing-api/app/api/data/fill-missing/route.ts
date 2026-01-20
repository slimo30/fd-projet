import { type NextRequest, NextResponse } from "next/server"

export async function POST(request: NextRequest) {
  try {
    const data = await request.json()

    // Forward the request to the Flask API
    const response = await fetch("http://127.0.0.1:5001/data/fill-missing", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(data),
    })

    if (!response.ok) {
      throw new Error(`Flask API error: ${response.statusText}`)
    }

    const result = await response.json()
    return NextResponse.json(result)
  } catch (error) {
    console.error("Error filling missing values:", error)
    return NextResponse.json({ error: "Failed to fill missing values" }, { status: 500 })
  }
}
