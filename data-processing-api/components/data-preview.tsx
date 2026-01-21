"use client";

import { useState, useEffect } from "react";
import { Loader2, Check, CheckSquare, Square, Save } from "lucide-react";
import { toast } from "sonner";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
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
import { ScrollArea, ScrollBar } from "@/components/ui/scroll-area";
import { dataApi, ApiError } from "@/lib/api";

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

  // Save functionality state
  const [isSaving, setIsSaving] = useState(false);
  const [savePath, setSavePath] = useState("");
  const [isSavingData, setIsSavingData] = useState(false);

  const handleSaveData = async () => {
    if (!savePath) {
      setError("Please enter a filename");
      return;
    }

    // Ensure .csv extension
    const filename = savePath.endsWith('.csv') ? savePath : `${savePath}.csv`;
    // Construct full path
    const directory = path.substring(0, path.lastIndexOf('/') + 1);
    const fullPath = directory + filename;

    setIsSavingData(true);
    setError(null);

    try {
      const result = await dataApi.saveData(fullPath);
      toast.success("Data saved successfully", {
        description: `Saved to ${result.path}`
      });
      setSavePath("");
      setIsSaving(false); // Close save section on success
    } catch (error) {
      console.error("Error saving data:", error);
      if (error instanceof ApiError) {
        setError(`Failed to save data: ${error.message}`);
      } else {
        setError(error instanceof Error ? error.message : "An unknown error occurred");
      }
    } finally {
      setIsSavingData(false);
    }
  };

  useEffect(() => {
    const fetchData = async () => {
      setLoading(true);
      setError(null);

      try {
        // Fetch representative rows with indices
        const result = await dataApi.getDataHead(path);
        setData(result || []);
        setSelectedColumns(columns);
      } catch (error) {
        console.error("Error fetching data:", error);
        if (error instanceof ApiError) {
          setError(`Failed to fetch data: ${error.message}`);
        } else {
          setError(error instanceof Error ? error.message : "An unknown error occurred");
        }
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
      const result = await dataApi.selectColumns(path, selectedColumns);

      // The Flask API doesn't change the path, so we use the same path
      onColumnsSelected(path);
    } catch (error) {
      console.error("Error applying column selection:", error);
      if (error instanceof ApiError) {
        setError(`Failed to apply selection: ${error.message}`);
      } else {
        setError(error instanceof Error ? error.message : "An unknown error occurred");
      }
    } finally {
      setIsApplying(false);
      setIsSelecting(false);
    }
  };

  const handleSelectAll = () => {
    setSelectedColumns(columns);
  };

  const handleDeselectAll = () => {
    setSelectedColumns([]);
  };

  const allSelected = selectedColumns.length === columns.length;
  const someSelected = selectedColumns.length > 0 && selectedColumns.length < columns.length;

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
          onClick={() => window.location.reload()}
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
      <div className="space-y-2">
        <div className="flex justify-between items-center">
          <div>
            <h3 className="text-lg font-semibold">Data Preview</h3>
            <p className="text-sm text-gray-500">
              Review your data and select columns for analysis
            </p>
          </div>
          <div className="flex gap-2">
            <Button
              variant={isSaving ? "default" : "outline"}
              onClick={() => {
                setIsSaving(!isSaving);
                setIsSelecting(false); // Close selection if open
              }}
            >
              <Save className="mr-2 h-4 w-4" />
              {isSaving ? "Cancel Save" : "Save as CSV"}
            </Button>
            <Button
              variant={isSelecting ? "default" : "outline"}
              onClick={() => {
                setIsSelecting(!isSelecting);
                setIsSaving(false); // Close save if open
              }}
            >
              {isSelecting ? "Cancel Selection" : "Select Columns"}
            </Button>
          </div>
        </div>

        {isSaving && (
          <div className="bg-green-50 border border-green-200 rounded-md p-4 space-y-4 animate-in slide-in-from-top-2">
            <h4 className="font-medium text-green-900 flex items-center gap-2">
              <Save className="h-4 w-4" />
              Save Dataset
            </h4>
            <div className="flex gap-3 items-end">
              <div className="grid w-full max-w-sm items-center gap-1.5">
                <Label htmlFor="filename">Filename</Label>
                <Input
                  type="text"
                  id="filename"
                  placeholder="processed_data.csv"
                  value={savePath}
                  onChange={(e) => setSavePath(e.target.value)}
                  className="bg-white"
                />
              </div>
              <Button
                onClick={handleSaveData}
                disabled={isSavingData || !savePath}
              >
                {isSavingData ? (
                  <>
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                    Saving...
                  </>
                ) : (
                  "Save File"
                )}
              </Button>
            </div>
            <p className="text-xs text-green-700">
              This will save the current state of the dataset (including filtered columns) to a new file.
            </p>
          </div>
        )}

        {isSelecting && (
          <div className="flex items-center justify-between bg-blue-50 border border-blue-200 rounded-md p-3">
            <div className="flex items-center space-x-4">
              <span className="text-sm font-medium text-blue-900">
                {selectedColumns.length} of {columns.length} columns selected
              </span>
              <div className="flex space-x-2">
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={handleSelectAll}
                  disabled={allSelected}
                  className="h-8 text-xs"
                >
                  <CheckSquare className="mr-1 h-3 w-3" />
                  Select All
                </Button>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={handleDeselectAll}
                  disabled={selectedColumns.length === 0}
                  className="h-8 text-xs"
                >
                  <Square className="mr-1 h-3 w-3" />
                  Deselect All
                </Button>
              </div>
            </div>
          </div>
        )}

        {!isSelecting && (
          <div className="flex items-center justify-between text-sm text-gray-500">
            <span>
              Showing {data.length} representative rows × {columns.length} columns
            </span>
            {data.length === 40 && (
              <span className="text-xs text-gray-400 italic">
                (Evenly sampled across dataset)
              </span>
            )}
          </div>
        )}
      </div>

      <div className="rounded-md border">
        <ScrollArea className="h-[500px] w-full">
          <div className="min-w-full">
            <Table>
              <TableHeader>
                <TableRow>
                  {/* Row Index Column */}
                  <TableHead className="sticky left-0 z-20 w-20 text-center bg-gray-100 border-r font-semibold">
                    Row #
                  </TableHead>
                  {isSelecting && (
                    <TableHead className="sticky left-20 z-20 w-16 text-center bg-gray-50 border-r">
                      <Check className="h-4 w-4 mx-auto" />
                    </TableHead>
                  )}
                  {columns.map((column) => (
                    <TableHead
                      key={column}
                      className={`min-w-[150px] ${isSelecting ? "bg-gray-50" : ""}`}
                    >
                      {isSelecting ? (
                        <div className="flex items-center space-x-2">
                          <Checkbox
                            id={`column-${column}`}
                            checked={selectedColumns.includes(column)}
                            onCheckedChange={() => handleColumnToggle(column)}
                          />
                          <label
                            htmlFor={`column-${column}`}
                            className="text-sm font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70 cursor-pointer"
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
                    {/* Display original row index from dataset */}
                    <TableCell className="sticky left-0 z-10 text-center text-sm font-medium text-gray-600 bg-gray-50 border-r">
                      {row.__row_index__ !== undefined ? row.__row_index__ : rowIndex}
                    </TableCell>
                    {isSelecting && (
                      <TableCell className="sticky left-20 z-10 text-center text-xs text-gray-400 bg-white border-r">
                        {rowIndex + 1}
                      </TableCell>
                    )}
                    {columns.map((column) => (
                      <TableCell key={`${rowIndex}-${column}`} className="min-w-[150px]">
                        {row[column] !== null && row[column] !== undefined
                          ? String(row[column])
                          : <span className="text-gray-400 italic">N/A</span>}
                      </TableCell>
                    ))}
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </div>
          <ScrollBar orientation="horizontal" />
          <ScrollBar orientation="vertical" />
        </ScrollArea>
      </div>

      {isSelecting && (
        <div className="flex justify-between items-center pt-2 border-t">
          <p className="text-sm text-gray-600">
            {selectedColumns.length === 0 && "Select at least one column to continue"}
            {selectedColumns.length > 0 && `${selectedColumns.length} column${selectedColumns.length !== 1 ? 's' : ''} will be used for analysis`}
          </p>
          <div className="flex space-x-2">
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
        </div>
      )}
    </div>
  );
}
