"use client";

import { useState } from "react";
import { toast } from "sonner";
import { Loader2, Info, HelpCircle, Save } from "lucide-react";
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
import { Slider } from "@/components/ui/slider";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
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
  const [knnK, setKnnK] = useState<number>(5);
  const [optimizationPlot, setOptimizationPlot] = useState<string | null>(null);
  const [isGeneratingPlot, setIsGeneratingPlot] = useState(false);
  const [suggestedK, setSuggestedK] = useState<number | null>(null);

  // Normalization state
  const [selectedColumnsForNormalization, setSelectedColumnsForNormalization] =
    useState<string[]>([]);
  const [normalizationMethod, setNormalizationMethod] =
    useState<string>("zscore");

  // Categorical to numerical state
  const [selectedColumnsForCategorical, setSelectedColumnsForCategorical] =
    useState<string[]>([]);
  const [encodingMethod, setEncodingMethod] = useState<string>("label");
  const [savePath, setSavePath] = useState<string>("");

  const handleSaveData = async () => {
    if (!savePath) {
      setError("Please enter a filename");
      return;
    }

    // Ensure .csv extension
    const filename = savePath.endsWith('.csv') ? savePath : `${savePath}.csv`;
    // Construct full path - simplistic approach, ideally handled by backend
    // but here we just pass the filename assuming backend handles the directory
    // or we construct a path in the same directory as the current file
    const directory = path.substring(0, path.lastIndexOf('/') + 1);
    const fullPath = directory + filename;

    setIsProcessing(true);
    setError(null);

    try {
      const result = await dataApi.saveData(fullPath);
      toast.success("Data saved successfully", {
        description: `Saved to ${result.path}`
      });
      setSavePath("");
    } catch (error) {
      console.error("Error saving data:", error);
      if (error instanceof ApiError) {
        setError(`Failed to save data: ${error.message}`);
      } else {
        setError(error instanceof Error ? error.message : "An unknown error occurred");
      }
    } finally {
      setIsProcessing(false);
    }
  };

  const handleFillMissingValues = async () => {
    if (selectedColumnsForMissing.length === 0) {
      setError("Please select at least one column");
      return;
    }

    setIsProcessing(true);
    setError(null);

    try {
      // Create fill strategy object
      const fill: Record<string, string | "mean" | "median" | "mode"> = {};
      selectedColumnsForMissing.forEach((column) => {
        if (fillStrategy === "knn") {
          fill[column] = `knn:${knnK}`;
        } else {
          fill[column] = fillStrategy as "mean" | "median" | "mode";
        }
      });

      await dataApi.fillMissing(
        path,
        fill as Record<string, "mean" | "median" | "mode">
      );

      // The Flask API doesn't change the path, so we use the same path
      onProcessingComplete(path);
    } catch (error) {
      console.error("Error filling missing values:", error);
      if (error instanceof ApiError) {
        setError(`Failed to fill missing values: ${error.message}`);
      } else {
        setError(
          error instanceof Error ? error.message : "An unknown error occurred"
        );
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
        normalizationMethod as "zscore" | "minmax",
        selectedColumnsForNormalization
      );

      // The Flask API doesn't change the path, so we use the same path
      onProcessingComplete(path);
    } catch (error) {
      console.error("Error normalizing data:", error);
      if (error instanceof ApiError) {
        setError(`Failed to normalize data: ${error.message}`);
      } else {
        setError(
          error instanceof Error ? error.message : "An unknown error occurred"
        );
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
        encodingMethod as "label" | "onehot",
        selectedColumnsForCategorical
      );

      // The Flask API doesn't change the path, so we use the same path
      onProcessingComplete(path);
    } catch (error) {
      console.error("Error converting categorical data:", error);
      if (error instanceof ApiError) {
        setError(`Failed to convert categorical data: ${error.message}`);
      } else {
        setError(
          error instanceof Error ? error.message : "An unknown error occurred"
        );
      }
    } finally {
      setIsProcessing(false);
    }
  };

  const handleGenerateKNNPlot = async () => {
    if (selectedColumnsForMissing.length === 0) {
      setError("Please select at least one column");
      return;
    }

    // Use the first selected column for optimization
    const column = selectedColumnsForMissing[0];

    setIsGeneratingPlot(true);
    setError(null);

    try {
      const result = await dataApi.getKNNOptimization(path, column, 20);

      setOptimizationPlot(result.plot_url);
      setSuggestedK(result.optimal_k);
      setKnnK(result.optimal_k); // Auto-set to optimal K

      toast.success(`Optimal K found: ${result.optimal_k}`, {
        description: `RMSE: ${result.min_error.toFixed(4)}`,
      });
    } catch (error) {
      console.error("Error generating KNN plot:", error);
      if (error instanceof ApiError) {
        setError(`Failed to generate plot: ${error.message}`);
      } else {
        setError(error instanceof Error ? error.message : "An unknown error occurred");
      }
    } finally {
      setIsGeneratingPlot(false);
    }
  };

  // Helper function to get K recommendation
  const getKRecommendation = (k: number) => {
    if (k < 3)
      return { text: "Too small - may overfit to noise", color: "text-red-600" };
    if (k >= 3 && k <= 5)
      return { text: "Good choice for most datasets", color: "text-green-600" };
    if (k > 5 && k <= 10)
      return {
        text: "Balanced - good for larger datasets",
        color: "text-blue-600",
      };
    if (k > 10 && k <= 15)
      return {
        text: "Conservative - smoother imputation",
        color: "text-yellow-600",
      };
    return {
      text: "High K - may oversimplify patterns",
      color: "text-orange-600",
    };
  };

  const recommendation = getKRecommendation(knnK);

  return (
    <div className="space-y-4">
      <Tabs value={activeTab} onValueChange={setActiveTab}>
        <TabsList className="grid grid-cols-4">
          <TabsTrigger value="missing">Fill Missing Values</TabsTrigger>
          <TabsTrigger value="normalize">Normalize Data</TabsTrigger>
          <TabsTrigger value="categorical">
            Categorical to Numerical
          </TabsTrigger>
          <TabsTrigger value="save">Save Data</TabsTrigger>
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
                  <div className="flex items-center space-x-2">
                    <RadioGroupItem value="knn" id="knn" />
                    <Label htmlFor="knn" className="flex items-center gap-2">
                      KNN Imputer (numeric columns)
                      <TooltipProvider>
                        <Tooltip>
                          <TooltipTrigger asChild>
                            <HelpCircle className="h-3 w-3 text-muted-foreground" />
                          </TooltipTrigger>
                          <TooltipContent className="max-w-xs">
                            <p className="text-xs">
                              Uses K nearest neighbors to predict missing values
                              based on similar rows. More accurate than simple
                              mean/median for correlated data.
                            </p>
                          </TooltipContent>
                        </Tooltip>
                      </TooltipProvider>
                    </Label>
                  </div>
                </RadioGroup>
              </div>

              {/* KNN K Parameter Selection */}
              {fillStrategy === "knn" && (
                <div className="space-y-3 border rounded-lg p-4 bg-muted/30">
                  <div className="flex items-center justify-between">
                    <Label htmlFor="knn-k" className="font-medium">
                      Number of Neighbors (K)
                    </Label>
                    <span className="text-2xl font-bold text-primary">
                      {knnK}
                    </span>
                  </div>

                  <Slider
                    id="knn-k"
                    min={1}
                    max={20}
                    step={1}
                    value={[knnK]}
                    onValueChange={(value) => setKnnK(value[0])}
                    className="w-full"
                  />

                  <div className="flex justify-between text-xs text-muted-foreground">
                    <span>1 (Less smooth)</span>
                    <span>20 (More smooth)</span>
                  </div>

                  {/* Recommendation */}
                  <div
                    className={`flex items-start gap-2 p-3 rounded-md bg-background border ${recommendation.color === "text-green-600"
                        ? "border-green-200 bg-green-50"
                        : recommendation.color === "text-blue-600"
                          ? "border-blue-200 bg-blue-50"
                          : recommendation.color === "text-yellow-600"
                            ? "border-yellow-200 bg-yellow-50"
                            : "border-gray-200"
                      }`}
                  >
                    <Info
                      className={`h-4 w-4 mt-0.5 ${recommendation.color}`}
                    />
                    <div className="flex-1">
                      <p
                        className={`text-sm font-medium ${recommendation.color}`}
                      >
                        {recommendation.text}
                      </p>
                      <p className="text-xs text-muted-foreground mt-1">
                        {knnK <= 5 &&
                          "Lower K captures local patterns but may be sensitive to noise."}
                        {knnK > 5 &&
                          knnK <= 10 &&
                          "Moderate K balances between capturing patterns and avoiding noise."}
                        {knnK > 10 &&
                          "Higher K creates smoother estimates but may miss local variations."}
                      </p>
                    </div>
                  </div>

                  {/* Quick presets */}
                  <div className="space-y-2">
                    <p className="text-xs font-medium text-muted-foreground">
                      Quick Presets:
                    </p>
                    <div className="flex gap-2">
                      <Button
                        type="button"
                        variant={knnK === 3 ? "default" : "outline"}
                        size="sm"
                        onClick={() => setKnnK(3)}
                        className="flex-1 text-xs"
                      >
                        K=3 (Sensitive)
                      </Button>
                      <Button
                        type="button"
                        variant={knnK === 5 ? "default" : "outline"}
                        size="sm"
                        onClick={() => setKnnK(5)}
                        className="flex-1 text-xs"
                      >
                        K=5 (Balanced)
                      </Button>
                      <Button
                        type="button"
                        variant={knnK === 10 ? "default" : "outline"}
                        size="sm"
                        onClick={() => setKnnK(10)}
                        className="flex-1 text-xs"
                      >
                        K=10 (Smooth)
                      </Button>
                    </div>
                  </div>

                  {/* Help text */}
                  <div className="text-xs text-muted-foreground bg-background rounded p-2 border">
                    <strong>How to choose K:</strong>
                    <ul className="list-disc list-inside mt-1 space-y-0.5">
                      <li>Small dataset (&lt;100 rows): K=3-5</li>
                      <li>Medium dataset (100-1000 rows): K=5-10</li>
                      <li>Large dataset (&gt;1000 rows): K=10-15</li>
                      <li>Rule of thumb: K = √(number of rows)</li>
                    </ul>
                  </div>

                  {/* Generate KNN Optimization Plot */}
                  <div className="space-y-2">
                    <Button
                      onClick={handleGenerateKNNPlot}
                      disabled={isGeneratingPlot || selectedColumnsForMissing.length === 0}
                      className="w-full"
                    >
                      {isGeneratingPlot ? (
                        <>
                          <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                          Generating Plot...
                        </>
                      ) : (
                        "Generate KNN Optimization Plot"
                      )}
                    </Button>
                    {optimizationPlot && (
                      <div className="space-y-2">
                        <Label>Optimization Plot</Label>
                        <img
                          src={optimizationPlot}
                          alt="KNN Optimization Plot"
                          className="w-full rounded-md border"
                        />
                        {suggestedK && (
                          <p className="text-sm text-muted-foreground">
                            Suggested K: <strong>{suggestedK}</strong>
                          </p>
                        )}
                      </div>
                    )}
                  </div>
                </div>
              )}

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

        <TabsContent value="save">
          <Card>
            <CardHeader>
              <CardTitle>Save Processed Data</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="space-y-2">
                <Label htmlFor="filename">Filename</Label>
                <div className="flex gap-2">
                  <div className="relative flex-1">
                    <input
                      id="filename"
                      type="text"
                      className="flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background file:border-0 file:bg-transparent file:text-sm file:font-medium placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50"
                      placeholder="processed_data.csv"
                      value={savePath}
                      onChange={(e) => setSavePath(e.target.value)}
                    />
                  </div>
                  <Button
                    onClick={handleSaveData}
                    disabled={isProcessing || !savePath}
                  >
                    {isProcessing ? (
                      <>
                        <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                        Saving...
                      </>
                    ) : (
                      <>
                        <Save className="mr-2 h-4 w-4" />
                        Save
                      </>
                    )}
                  </Button>
                </div>
                <p className="text-sm text-muted-foreground">
                  Save your processed dataset to a new CSV file.
                </p>
              </div>

              {error && (
                <div className="bg-red-50 text-red-500 p-3 rounded-md text-sm">
                  {error}
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}
