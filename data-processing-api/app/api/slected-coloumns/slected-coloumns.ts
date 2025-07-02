import { NextRequest, NextResponse } from "next/server";

interface SelectedColumnsResponse {
  selected_columns: string[];
  message?: string;
}

let selectedColumns: string[] = []; // You can fetch or manage these columns however you prefer

// This function will be called when a GET request is made to the endpoint
export async function GET(req: NextRequest) {
  if (selectedColumns.length > 0) {
    return NextResponse.json({ selected_columns: selectedColumns });
  } else {
    return NextResponse.json({ message: "No columns selected" });
  }
}
