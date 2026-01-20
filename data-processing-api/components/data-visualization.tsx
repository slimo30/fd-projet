"use client";

import { useState } from "react";
import { Loader2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Checkbox } from "@/components/ui/checkbox";
import { Label } from "@/components/ui/label";
import { plottingApi, ApiError, getImageUrl } from "@/lib/api";

interface DataVisualizationProps {
  path: string;
  columns: string[];
}

export default function DataVisualization({
  path,
  columns,
}: DataVisualizationProps) {
  const [activeTab, setActiveTab] = useState("scatter");
  const [isGenerating, setIsGenerating] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [plotUrl, setPlotUrl] = useState<string | null>(null);

  // Scatter plot state
  const [xColumn, setXColumn] = useState<string>("");
  const [yColumn, setYColumn] = useState<string>("");

  // Box plot state
  const [selectedColumnsForBox, setSelectedColumnsForBox] = useState<string[]>(
    []
  );

  const handleGenerateScatterPlot = async () => {
    if (!xColumn || !yColumn) {
      setError("Please select both X and Y columns");
      return;
    }

    setIsGenerating(true);
    setError(null);
    setPlotUrl(null);

    try {
      const result = await plottingApi.scatterPlot(path, xColumn, yColumn);

      if (result.plot_url) {
        setPlotUrl(getImageUrl(result.plot_url));
      } else {
        throw new Error("No plot URL returned from server");
      }
    } catch (error) {
      console.error("Error generating scatter plot:", error);
      if (error instanceof ApiError) {
        setError(`Failed to generate scatter plot: ${error.message}`);
      } else {
        setError(error instanceof Error ? error.message : "An unknown error occurred");
      }
    } finally {
      setIsGenerating(false);
    }
  };

  const handleGenerateBoxPlot = async () => {
    if (selectedColumnsForBox.length === 0) {
      setError("Please select at least one column");
      return;
    }

    setIsGenerating(true);
    setError(null);
    setPlotUrl(null);

    try {
      const result = await plottingApi.boxPlot(path, selectedColumnsForBox);

      if (result.plot_url) {
        setPlotUrl(getImageUrl(result.plot_url));
      } else {
        throw new Error("No plot URL returned from server");
      }
    } catch (error) {
      console.error("Error generating box plot:", error);
      if (error instanceof ApiError) {
        setError(`Failed to generate box plot: ${error.message}`);
      } else {
        setError(error instanceof Error ? error.message : "An unknown error occurred");
      }
    } finally {
      setIsGenerating(false);
    }
  };

  return (
    <div className="space-y-4">
      <Tabs value={activeTab} onValueChange={setActiveTab}>
        <TabsList className="grid grid-cols-2">
          <TabsTrigger value="scatter">Scatter Plot</TabsTrigger>
          <TabsTrigger value="box">Box Plot</TabsTrigger>
        </TabsList>

        <TabsContent value="scatter">
          <Card>
            <CardHeader>
              <CardTitle>Scatter Plot</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="space-y-2">
                  <Label>X-Axis Column</Label>
                  <Select value={xColumn} onValueChange={setXColumn}>
                    <SelectTrigger>
                      <SelectValue placeholder="Select X column" />
                    </SelectTrigger>
                    <SelectContent>
                      {columns.map((column) => (
                        <SelectItem key={`x-${column}`} value={column}>
                          {column}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>

                <div className="space-y-2">
                  <Label>Y-Axis Column</Label>
                  <Select value={yColumn} onValueChange={setYColumn}>
                    <SelectTrigger>
                      <SelectValue placeholder="Select Y column" />
                    </SelectTrigger>
                    <SelectContent>
                      {columns.map((column) => (
                        <SelectItem key={`y-${column}`} value={column}>
                          {column}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
              </div>

              {error && (
                <div className="bg-red-50 text-red-500 p-3 rounded-md text-sm">
                  {error}
                </div>
              )}

              <Button
                onClick={handleGenerateScatterPlot}
                disabled={isGenerating || !xColumn || !yColumn}
                className="w-full"
              >
                {isGenerating ? (
                  <>
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                    Generating...
                  </>
                ) : (
                  "Generate Scatter Plot"
                )}
              </Button>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="box">
          <Card>
            <CardHeader>
              <CardTitle>Box Plot</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="space-y-2">
                <Label>Select Columns (numeric only)</Label>
                <div className="grid grid-cols-2 md:grid-cols-3 gap-2 border rounded-md p-3 max-h-40 overflow-y-auto">
                  {columns.map((column) => (
                    <div key={column} className="flex items-center space-x-2">
                      <Checkbox
                        id={`box-${column}`}
                        checked={selectedColumnsForBox.includes(column)}
                        onCheckedChange={(checked) => {
                          if (checked) {
                            setSelectedColumnsForBox([
                              ...selectedColumnsForBox,
                              column,
                            ]);
                          } else {
                            setSelectedColumnsForBox(
                              selectedColumnsForBox.filter(
                                (col) => col !== column
                              )
                            );
                          }
                        }}
                      />
                      <label
                        htmlFor={`box-${column}`}
                        className="text-sm font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70"
                      >
                        {column}
                      </label>
                    </div>
                  ))}
                </div>
              </div>

              {error && (
                <div className="bg-red-50 text-red-500 p-3 rounded-md text-sm">
                  {error}
                </div>
              )}

              <Button
                onClick={handleGenerateBoxPlot}
                disabled={isGenerating || selectedColumnsForBox.length === 0}
                className="w-full"
              >
                {isGenerating ? (
                  <>
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                    Generating...
                  </>
                ) : (
                  "Generate Box Plot"
                )}
              </Button>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>

      {plotUrl && (
        <div className="mt-4 border rounded-md p-4">
          <h3 className="text-lg font-medium mb-2">Plot Preview</h3>
          <div className="flex justify-center">
            {/* key={plotUrl} forces React to re-render the img element when URL changes */}
            <img
              key={plotUrl}
              src={plotUrl || "/placeholder.svg"}
              alt="Data visualization"
              className="max-w-full h-auto rounded-md"
              style={{ maxHeight: "400px" }}
              onError={(e) => {
                console.error("Error loading image:", plotUrl);
                e.currentTarget.style.display = 'none';
              }}
            />
          </div>
          <div className="mt-2 flex justify-center">
            <Button
              variant="outline"
              onClick={() => window.open(plotUrl, "_blank")}
            >
              Open in New Tab
            </Button>
          </div>
        </div>
      )}
    </div>
  );
}
