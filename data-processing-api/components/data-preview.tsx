"use client";

import { useState, useEffect } from "react";
import { Loader2, Check } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { ScrollArea } from "@/components/ui/scroll-area";

interface DataPreviewProps {
  path: string;
  columns: string[];
  onColumnsSelected: (newPath: string) => void;
}

interface DataRow {
  [key: string]: any;
}

export default function DataPreview({
  path,
  columns,
  onColumnsSelected,
}: DataPreviewProps) {
  const [data, setData] = useState<DataRow[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedColumns, setSelectedColumns] = useState<string[]>([]);
  const [isSelecting, setIsSelecting] = useState(false);
  const [isApplying, setIsApplying] = useState(false);

  useEffect(() => {
    const fetchData = async () => {
      setLoading(true);
      setError(null);

      try {
        const response = await fetch(`/api/data/head?path=${path}`);

        if (!response.ok) {
          throw new Error(`Failed to fetch data: ${response.statusText}`);
        }

        const result = await response.json();
        setData(result || []);
        setSelectedColumns(columns);
      } catch (error) {
        console.error("Error fetching data:", error);
        setError(
          error instanceof Error ? error.message : "An unknown error occurred"
        );
      } finally {
        setLoading(false);
      }
    };

    if (path) {
      fetchData();
    }
  }, [path, columns]);

  const handleColumnToggle = (column: string) => {
    setSelectedColumns((prev) => {
      if (prev.includes(column)) {
        return prev.filter((col) => col !== column);
      } else {
        return [...prev, column];
      }
    });
  };

  const handleApplySelection = async () => {
    if (selectedColumns.length === 0) {
      setError("Please select at least one column");
      return;
    }

    setIsApplying(true);
    setError(null);

    try {
      const response = await fetch("/api/data/select-columns", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          path,
          columns: selectedColumns,
        }),
      });

      if (!response.ok) {
        throw new Error(`Failed to apply selection: ${response.statusText}`);
      }

      const result = await response.json();

      // The Flask API doesn't change the path, so we use the same path
      onColumnsSelected(path);
    } catch (error) {
      console.error("Error applying column selection:", error);
      setError(
        error instanceof Error ? error.message : "An unknown error occurred"
      );
    } finally {
      setIsApplying(false);
      setIsSelecting(false);
    }
  };

  if (loading) {
    return (
      <div className="flex justify-center items-center h-40">
        <Loader2 className="h-8 w-8 animate-spin text-primary" />
      </div>
    );
  }

  if (error) {
    return (
      <div className="bg-red-50 text-red-500 p-4 rounded-md">
        <p>{error}</p>
        <Button
          variant="outline"
          className="mt-2"
          onClick={() => setError(null)}
        >
          Retry
        </Button>
      </div>
    );
  }

  if (data.length === 0) {
    return (
      <div className="text-center p-4">
        <p>No data available</p>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <div className="flex justify-between items-center">
        <h3 className="text-lg font-medium">Data Preview</h3>
        <Button
          variant={isSelecting ? "default" : "outline"}
          onClick={() => setIsSelecting(!isSelecting)}
        >
          {isSelecting ? "Cancel Selection" : "Select Columns"}
        </Button>
      </div>

      <ScrollArea className="h-[400px] rounded-md border">
        <Table>
          <TableHeader>
            <TableRow>
              {isSelecting && (
                <TableHead className="w-12 text-center">
                  <Check className="h-4 w-4 mx-auto" />
                </TableHead>
              )}
              {columns.map((column) => (
                <TableHead key={column}>
                  {isSelecting ? (
                    <div className="flex items-center space-x-2">
                      <Checkbox
                        id={`column-${column}`}
                        checked={selectedColumns.includes(column)}
                        onCheckedChange={() => handleColumnToggle(column)}
                      />
                      <label
                        htmlFor={`column-${column}`}
                        className="text-sm font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70"
                      >
                        {column}
                      </label>
                    </div>
                  ) : (
                    column
                  )}
                </TableHead>
              ))}
            </TableRow>
          </TableHeader>
          <TableBody>
            {data.map((row, rowIndex) => (
              <TableRow key={rowIndex}>
                {isSelecting && (
                  <TableCell className="text-center">{rowIndex + 1}</TableCell>
                )}
                {columns.map((column) => (
                  <TableCell key={`${rowIndex}-${column}`}>
                    {row[column] !== null && row[column] !== undefined
                      ? String(row[column])
                      : "N/A"}
                  </TableCell>
                ))}
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </ScrollArea>

      {isSelecting && (
        <div className="flex justify-end space-x-2">
          <Button variant="outline" onClick={() => setIsSelecting(false)}>
            Cancel
          </Button>
          <Button
            onClick={handleApplySelection}
            disabled={isApplying || selectedColumns.length === 0}
          >
            {isApplying ? (
              <>
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                Applying...
              </>
            ) : (
              "Apply Selection"
            )}
          </Button>
        </div>
      )}
    </div>
  );
}
