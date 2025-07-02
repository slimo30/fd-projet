import { type NextRequest, NextResponse } from "next/server"

// This proxy helps avoid CORS issues when the frontend is running on a different domain/port than the API
export async function POST(request: NextRequest) {
  const url = request.nextUrl.searchParams.get("url")

  if (!url) {
    return NextResponse.json({ error: "Missing URL parameter" }, { status: 400 })
  }

  try {
    const apiUrl = `http://localhost:5000${url}`
    const contentType = request.headers.get("Content-Type")

    // Forward the request to the Python API
    const response = await fetch(apiUrl, {
      method: "POST",
      headers: {
        "Content-Type": contentType || "application/json",
      },
      body: request.body,
    })

    const data = await response.json()
    return NextResponse.json(data, { status: response.status })
  } catch (error) {
    console.error("Proxy error:", error)
    return NextResponse.json({ error: "Failed to connect to the clustering API" }, { status: 500 })
  }
}

export async function GET(request: NextRequest) {
  const url = request.nextUrl.searchParams.get("url")

  if (!url) {
    return NextResponse.json({ error: "Missing URL parameter" }, { status: 400 })
  }

  try {
    const apiUrl = `http://localhost:5000${url}`

    // Forward the request to the Python API
    const response = await fetch(apiUrl)
    const data = await response.json()

    return NextResponse.json(data, { status: response.status })
  } catch (error) {
    console.error("Proxy error:", error)
    return NextResponse.json({ error: "Failed to connect to the clustering API" }, { status: 500 })
  }
}
