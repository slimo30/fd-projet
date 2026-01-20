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
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group";
import { dataApi, ApiError } from "@/lib/api";

interface DataProcessingProps {
  path: string;
  columns: string[];
  onProcessingComplete: (newPath: string) => void;
}

export default function DataProcessing({
  path,
  columns,
  onProcessingComplete,
}: DataProcessingProps) {
  const [activeTab, setActiveTab] = useState("missing");
  const [isProcessing, setIsProcessing] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Missing values state
  const [selectedColumnsForMissing, setSelectedColumnsForMissing] = useState<
    string[]
  >([]);
  const [fillStrategy, setFillStrategy] = useState<string>("mean");

  // Normalization state
  const [selectedColumnsForNormalization, setSelectedColumnsForNormalization] =
    useState<string[]>([]);
  const [normalizationMethod, setNormalizationMethod] =
    useState<string>("zscore");

  // Categorical to numerical state
  const [selectedColumnsForCategorical, setSelectedColumnsForCategorical] =
    useState<string[]>([]);
  const [encodingMethod, setEncodingMethod] = useState<string>("label");

  const handleFillMissingValues = async () => {
    if (selectedColumnsForMissing.length === 0) {
      setError("Please select at least one column");
      return;
    }

    setIsProcessing(true);
    setError(null);

    try {
      // Create fill strategy object
      const fill: Record<string, 'mean' | 'median' | 'mode'> = {};
      selectedColumnsForMissing.forEach((column) => {
        fill[column] = fillStrategy as 'mean' | 'median' | 'mode';
      });

      await dataApi.fillMissing(path, fill);

      // The Flask API doesn't change the path, so we use the same path
      onProcessingComplete(path);
    } catch (error) {
      console.error("Error filling missing values:", error);
      if (error instanceof ApiError) {
        setError(`Failed to fill missing values: ${error.message}`);
      } else {
        setError(error instanceof Error ? error.message : "An unknown error occurred");
      }
    } finally {
      setIsProcessing(false);
    }
  };

  const handleNormalizeData = async () => {
    if (selectedColumnsForNormalization.length === 0) {
      setError("Please select at least one column");
      return;
    }

    setIsProcessing(true);
    setError(null);

    try {
      await dataApi.normalize(
        path,
        normalizationMethod as 'zscore' | 'minmax',
        selectedColumnsForNormalization
      );

      // The Flask API doesn't change the path, so we use the same path
      onProcessingComplete(path);
    } catch (error) {
      console.error("Error normalizing data:", error);
      if (error instanceof ApiError) {
        setError(`Failed to normalize data: ${error.message}`);
      } else {
        setError(error instanceof Error ? error.message : "An unknown error occurred");
      }
    } finally {
      setIsProcessing(false);
    }
  };

  const handleCategoricalToNumerical = async () => {
    if (selectedColumnsForCategorical.length === 0) {
      setError("Please select at least one column");
      return;
    }

    setIsProcessing(true);
    setError(null);

    try {
      await dataApi.categoricalToNumerical(
        path,
        encodingMethod as 'label' | 'onehot',
        selectedColumnsForCategorical
      );

      // The Flask API doesn't change the path, so we use the same path
      onProcessingComplete(path);
    } catch (error) {
      console.error("Error converting categorical data:", error);
      if (error instanceof ApiError) {
        setError(`Failed to convert categorical data: ${error.message}`);
      } else {
        setError(error instanceof Error ? error.message : "An unknown error occurred");
      }
    } finally {
      setIsProcessing(false);
    }
  };

  return (
    <div className="space-y-4">
      <Tabs value={activeTab} onValueChange={setActiveTab}>
        <TabsList className="grid grid-cols-3">
          <TabsTrigger value="missing">Fill Missing Values</TabsTrigger>
          <TabsTrigger value="normalize">Normalize Data</TabsTrigger>
          <TabsTrigger value="categorical">
            Categorical to Numerical
          </TabsTrigger>
        </TabsList>

        <TabsContent value="missing">
          <Card>
            <CardHeader>
              <CardTitle>Fill Missing Values</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="space-y-2">
                <Label>Select Columns</Label>
                <div className="grid grid-cols-2 md:grid-cols-3 gap-2 border rounded-md p-3 max-h-40 overflow-y-auto">
                  {columns.map((column) => (
                    <div key={column} className="flex items-center space-x-2">
                      <Checkbox
                        id={`missing-${column}`}
                        checked={selectedColumnsForMissing.includes(column)}
                        onCheckedChange={(checked) => {
                          if (checked) {
                            setSelectedColumnsForMissing([
                              ...selectedColumnsForMissing,
                              column,
                            ]);
                          } else {
                            setSelectedColumnsForMissing(
                              selectedColumnsForMissing.filter(
                                (col) => col !== column
                              )
                            );
                          }
                        }}
                      />
                      <label
                        htmlFor={`missing-${column}`}
                        className="text-sm font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70"
                      >
                        {column}
                      </label>
                    </div>
                  ))}
                </div>
              </div>

              <div className="space-y-2">
                <Label>Fill Strategy</Label>
                <RadioGroup
                  value={fillStrategy}
                  onValueChange={setFillStrategy}
                  className="flex flex-col space-y-1"
                >
                  <div className="flex items-center space-x-2">
                    <RadioGroupItem value="mean" id="mean" />
                    <Label htmlFor="mean">Mean (for numeric columns)</Label>
                  </div>
                  <div className="flex items-center space-x-2">
                    <RadioGroupItem value="median" id="median" />
                    <Label htmlFor="median">Median (for numeric columns)</Label>
                  </div>
                  <div className="flex items-center space-x-2">
                    <RadioGroupItem value="mode" id="mode" />
                    <Label htmlFor="mode">Mode (for any column type)</Label>
                  </div>
                </RadioGroup>
              </div>

              {error && (
                <div className="bg-red-50 text-red-500 p-3 rounded-md text-sm">
                  {error}
                </div>
              )}

              <Button
                onClick={handleFillMissingValues}
                disabled={
                  isProcessing || selectedColumnsForMissing.length === 0
                }
                className="w-full"
              >
                {isProcessing ? (
                  <>
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                    Processing...
                  </>
                ) : (
                  "Fill Missing Values"
                )}
              </Button>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="normalize">
          <Card>
            <CardHeader>
              <CardTitle>Normalize Data</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="space-y-2">
                <Label>Select Columns (numeric only)</Label>
                <div className="grid grid-cols-2 md:grid-cols-3 gap-2 border rounded-md p-3 max-h-40 overflow-y-auto">
                  {columns.map((column) => (
                    <div key={column} className="flex items-center space-x-2">
                      <Checkbox
                        id={`normalize-${column}`}
                        checked={selectedColumnsForNormalization.includes(
                          column
                        )}
                        onCheckedChange={(checked) => {
                          if (checked) {
                            setSelectedColumnsForNormalization([
                              ...selectedColumnsForNormalization,
                              column,
                            ]);
                          } else {
                            setSelectedColumnsForNormalization(
                              selectedColumnsForNormalization.filter(
                                (col) => col !== column
                              )
                            );
                          }
                        }}
                      />
                      <label
                        htmlFor={`normalize-${column}`}
                        className="text-sm font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70"
                      >
                        {column}
                      </label>
                    </div>
                  ))}
                </div>
              </div>

              <div className="space-y-2">
                <Label>Normalization Method</Label>
                <Select
                  value={normalizationMethod}
                  onValueChange={setNormalizationMethod}
                >
                  <SelectTrigger>
                    <SelectValue placeholder="Select method" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="zscore">
                      Z-Score (Standard Scaling)
                    </SelectItem>
                    <SelectItem value="minmax">Min-Max Scaling</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              {error && (
                <div className="bg-red-50 text-red-500 p-3 rounded-md text-sm">
                  {error}
                </div>
              )}

              <Button
                onClick={handleNormalizeData}
                disabled={
                  isProcessing || selectedColumnsForNormalization.length === 0
                }
                className="w-full"
              >
                {isProcessing ? (
                  <>
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                    Processing...
                  </>
                ) : (
                  "Normalize Data"
                )}
              </Button>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="categorical">
          <Card>
            <CardHeader>
              <CardTitle>Convert Categorical to Numerical</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="space-y-2">
                <Label>Select Categorical Columns</Label>
                <div className="grid grid-cols-2 md:grid-cols-3 gap-2 border rounded-md p-3 max-h-40 overflow-y-auto">
                  {columns.map((column) => (
                    <div key={column} className="flex items-center space-x-2">
                      <Checkbox
                        id={`categorical-${column}`}
                        checked={selectedColumnsForCategorical.includes(column)}
                        onCheckedChange={(checked) => {
                          if (checked) {
                            setSelectedColumnsForCategorical([
                              ...selectedColumnsForCategorical,
                              column,
                            ]);
                          } else {
                            setSelectedColumnsForCategorical(
                              selectedColumnsForCategorical.filter(
                                (col) => col !== column
                              )
                            );
                          }
                        }}
                      />
                      <label
                        htmlFor={`categorical-${column}`}
                        className="text-sm font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70"
                      >
                        {column}
                      </label>
                    </div>
                  ))}
                </div>
              </div>

              <div className="space-y-2">
                <Label>Encoding Method</Label>
                <RadioGroup
                  value={encodingMethod}
                  onValueChange={setEncodingMethod}
                  className="flex flex-col space-y-1"
                >
                  <div className="flex items-center space-x-2">
                    <RadioGroupItem value="label" id="label" />
                    <Label htmlFor="label">
                      Label Encoding (single column)
                    </Label>
                  </div>
                  <div className="flex items-center space-x-2">
                    <RadioGroupItem value="onehot" id="onehot" />
                    <Label htmlFor="onehot">
                      One-Hot Encoding (multiple columns)
                    </Label>
                  </div>
                </RadioGroup>
              </div>

              {error && (
                <div className="bg-red-50 text-red-500 p-3 rounded-md text-sm">
                  {error}
                </div>
              )}

              <Button
                onClick={handleCategoricalToNumerical}
                disabled={
                  isProcessing || selectedColumnsForCategorical.length === 0
                }
                className="w-full"
              >
                {isProcessing ? (
                  <>
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                    Processing...
                  </>
                ) : (
                  "Convert Data"
                )}
              </Button>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}
