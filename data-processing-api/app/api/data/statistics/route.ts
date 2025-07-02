import { type NextRequest, NextResponse } from "next/server";

export async function GET(request: NextRequest) {
  console.log("\n=== Testing Get Statistics ===");

  try {
    // Extract the 'path' parameter from the query string
    const { searchParams } = new URL(request.url);
    const path = searchParams.get("path");

    // If no path is provided, return an error response
    if (!path) {
      console.error("No path provided");
      return NextResponse.json({ error: "No path provided" }, { status: 400 });
    }

    // Construct the Flask API URL
    const flaskUrl = `http://127.0.0.1:5000/data/statistics?path=${encodeURIComponent(
      path
    )}`;
    console.log(`Sending request to: ${flaskUrl}`);

    // Make the request to the Flask API
    const response = await fetch(flaskUrl);

    // Log the response status
    console.log(`Status Code: ${response.status}`);

    // If the response status is not OK (not in the 200-299 range), handle it
    if (!response.ok) {
      const text = await response.text(); // Read error response as text
      console.error("Flask server error response:", text);

      // Return the error message from Flask (JSON or plain text)
      return NextResponse.json(
        { error: "Failed to get data statistics", details: text },
        { status: response.status }
      );
    }

    // Parse the response as JSON
    const json = await response.json();
    console.log("Response JSON:", json);

    // Return the JSON response back to the client
    return NextResponse.json(json);
  } catch (error) {
    // Log any unexpected errors
    console.error("Error getting data statistics:", error);

    // Return a 500 error if an unexpected error occurs
    return NextResponse.json(
      { error: "Failed to get data statistics" },
      { status: 500 }
    );
  }
}
