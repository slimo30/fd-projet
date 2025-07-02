import { type NextRequest, NextResponse } from "next/server"

export async function GET(request: NextRequest) {
  try {
    // Get the path from the query parameters
    const { searchParams } = new URL(request.url)
    const path = searchParams.get("path")

    if (!path) {
      return NextResponse.json({ error: "No path provided" }, { status: 400 })
    }

    // Forward the request to the Flask API
    const response = await fetch(`http://127.0.0.1:5000/data/head?path=${encodeURIComponent(path)}`)

    if (!response.ok) {
      throw new Error(`Flask API error: ${response.statusText}`)
    }

    const data = await response.json()
    return NextResponse.json(data)
  } catch (error) {
    console.error("Error getting data head:", error)
    return NextResponse.json({ error: "Failed to get data head" }, { status: 500 })
  }
}
