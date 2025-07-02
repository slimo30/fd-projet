import { type NextRequest, NextResponse } from "next/server";

export async function POST(request: NextRequest) {
  try {
    // Create a sample CSV file
    const sampleFile = await createSampleData();

    // Upload the sample file to the Flask API
    const formData = new FormData();
    formData.append("file", sampleFile);

    const response = await fetch("http://127.0.0.1:5000/upload-data", {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      throw new Error(`Flask API error: ${response.statusText}`);
    }

    const data = await response.json();
    return NextResponse.json(data);
  } catch (error) {
    console.error("Error creating sample data:", error);
    return NextResponse.json(
      { error: "Failed to create sample data" },
      { status: 500 }
    );
  }
}

// Helper function to create a sample CSV file
async function createSampleData() {
  // Create sample data similar to the test script
  const headers = "id,category,value,status,rating\n";
  const rows = [];

  const categories = ["A", "B", "C", "D", "E"];
  const statuses = ["active", "inactive", "pending", "unknown"];

  for (let i = 1; i <= 100; i++) {
    // Ensure that the random data does not introduce invalid (empty or NaN) values
    const id = i;
    const category = categories[i % categories.length];
    const value = i;
    const status = statuses[i % statuses.length];
    const rating = (i % 5) + 1;

    rows.push(`${id},${category},${value},${status},${rating}`);
  }

  const csvContent = headers + rows.join("\n");

  // Create a file from the CSV content
  return new File([csvContent], "sample_data.csv", { type: "text/csv" });
}
