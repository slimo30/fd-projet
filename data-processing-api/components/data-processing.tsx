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
  onProcessingComplete: (newPath: string, newColumns?: string[]) => void;
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

  // Advanced state
  const [selectedColumnForBinarize, setSelectedColumnForBinarize] = useState<string>("");
  const [binarizeZeroGroup, setBinarizeZeroGroup] = useState<string>("");
  const [binarizeThreshold, setBinarizeThreshold] = useState<string>("");
  const [selectedColumnForOrdinal, setSelectedColumnForOrdinal] = useState<string>("");
  const [ordinalOrder, setOrdinalOrder] = useState<string>("");

  const handleSaveData = async () => {
    if (!savePath) {
      setError("Please enter a filename");
      return;
    }

    const filename = savePath.endsWith('.csv') ? savePath : `${savePath}.csv`;
    const lastSlashIndex = Math.max(path.lastIndexOf('/'), path.lastIndexOf('\\'));
    const directory = lastSlashIndex === -1 ? '' : path.substring(0, lastSlashIndex + 1);
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
      const fill: Record<string, string> = {};
      selectedColumnsForMissing.forEach((column) => {
        if (fillStrategy === "knn") {
          fill[column] = `knn:${knnK}`;
        } else {
          fill[column] = fillStrategy;
        }
      });

      await dataApi.fillMissing(path, fill as any);
      const { columns: newColumns } = await dataApi.getColumns(path);
      onProcessingComplete(path, newColumns);
    } catch (error) {
      console.error("Error filling missing values:", error);
      setError(error instanceof Error ? error.message : "An unknown error occurred");
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

      const { columns: newColumns } = await dataApi.getColumns(path);
      onProcessingComplete(path, newColumns);
    } catch (error) {
      console.error("Error normalizing data:", error);
      setError(error instanceof Error ? error.message : "An unknown error occurred");
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

      const { columns: newColumns } = await dataApi.getColumns(path);
      onProcessingComplete(path, newColumns);
    } catch (error) {
      console.error("Error converting categorical data:", error);
      setError(error instanceof Error ? error.message : "An unknown error occurred");
    } finally {
      setIsProcessing(false);
    }
  };

  const handleBinarize = async () => {
    if (!selectedColumnForBinarize) {
      setError("Please select a column to binarize");
      return;
    }

    setIsProcessing(true);
    setError(null);

    try {
      const zero_group = binarizeZeroGroup
        ? binarizeZeroGroup.split(',').map(s => s.trim())
        : undefined;
      const threshold = binarizeThreshold ? parseFloat(binarizeThreshold) : undefined;

      await dataApi.binarize(path, selectedColumnForBinarize, { zero_group, threshold });

      toast.success(`Column ${selectedColumnForBinarize} binarized`);
      const { columns: newColumns } = await dataApi.getColumns(path);
      onProcessingComplete(path, newColumns);
    } catch (error) {
      console.error("Error binarizing data:", error);
      setError(error instanceof Error ? error.message : "An unknown error occurred");
    } finally {
      setIsProcessing(false);
    }
  };

  const handleOrdinalMap = async () => {
    if (!selectedColumnForOrdinal || !ordinalOrder) {
      setError("Please select a column and provide the order");
      return;
    }

    setIsProcessing(true);
    setError(null);

    try {
      const order = ordinalOrder.split(',').map(s => s.trim());
      await dataApi.ordinalMap(path, selectedColumnForOrdinal, order);

      toast.success(`Column ${selectedColumnForOrdinal} ordinal mapped`);
      const { columns: newColumns } = await dataApi.getColumns(path);
      onProcessingComplete(path, newColumns);
    } catch (error) {
      console.error("Error ordinal mapping data:", error);
      setError(error instanceof Error ? error.message : "An unknown error occurred");
    } finally {
      setIsProcessing(false);
    }
  };

  const handleGenerateKNNPlot = async () => {
    if (selectedColumnsForMissing.length === 0) {
      setError("Please select at least one column");
      return;
    }

    const column = selectedColumnsForMissing[0];
    setIsGeneratingPlot(true);
    setError(null);

    try {
      const result = await dataApi.getKNNOptimization(path, column, 20);
      setOptimizationPlot(result.plot_url);
      setSuggestedK(result.optimal_k);
      setKnnK(result.optimal_k);
      toast.success(`Optimal K found: ${result.optimal_k}`);
    } catch (error) {
      console.error("Error generating KNN plot:", error);
      setError(error instanceof Error ? error.message : "An unknown error occurred");
    } finally {
      setIsGeneratingPlot(false);
    }
  };

  const getKRecommendation = (k: number) => {
    if (k < 3) return { text: "Too small", color: "text-red-600" };
    if (k <= 5) return { text: "Balanced", color: "text-green-600" };
    return { text: "Large K", color: "text-blue-600" };
  };

  const recommendation = getKRecommendation(knnK);

  return (
    <div className="space-y-4">
      <Tabs value={activeTab} onValueChange={setActiveTab}>
        <TabsList className="grid grid-cols-5">
          <TabsTrigger value="missing">Missing</TabsTrigger>
          <TabsTrigger value="normalize">Normalize</TabsTrigger>
          <TabsTrigger value="categorical">Categorical</TabsTrigger>
          <TabsTrigger value="advanced">Advanced</TabsTrigger>
          <TabsTrigger value="save">Save</TabsTrigger>
        </TabsList>

        <TabsContent value="missing">
          <Card>
            <CardHeader><CardTitle>Fill Missing Values</CardTitle></CardHeader>
            <CardContent className="space-y-4">
              <div className="space-y-2">
                <Label>Select Columns</Label>
                <div className="grid grid-cols-3 gap-2 border rounded-md p-3 max-h-40 overflow-y-auto">
                  {columns.map((column) => (
                    <div key={column} className="flex items-center space-x-2">
                      <Checkbox
                        id={`missing-${column}`}
                        checked={selectedColumnsForMissing.includes(column)}
                        onCheckedChange={(checked) => {
                          if (checked) setSelectedColumnsForMissing([...selectedColumnsForMissing, column]);
                          else setSelectedColumnsForMissing(selectedColumnsForMissing.filter(col => col !== column));
                        }}
                      />
                      <Label htmlFor={`missing-${column}`}>{column}</Label>
                    </div>
                  ))}
                </div>
              </div>

              <div className="space-y-2">
                <Label>Fill Strategy</Label>
                <RadioGroup value={fillStrategy} onValueChange={setFillStrategy}>
                  <div className="flex items-center space-x-2">
                    <RadioGroupItem value="mean" id="mean" /><Label htmlFor="mean">Mean</Label>
                  </div>
                  <div className="flex items-center space-x-2">
                    <RadioGroupItem value="median" id="median" /><Label htmlFor="median">Median</Label>
                  </div>
                  <div className="flex items-center space-x-2">
                    <RadioGroupItem value="mode" id="mode" /><Label htmlFor="mode">Mode</Label>
                  </div>
                  <div className="flex items-center space-x-2">
                    <RadioGroupItem value="knn" id="knn" /><Label htmlFor="knn">KNN Imputer</Label>
                  </div>
                </RadioGroup>
              </div>

              {fillStrategy === "knn" && (
                <div className="space-y-3 border rounded-lg p-4 bg-muted/30">
                  <div className="flex items-center justify-between">
                    <Label htmlFor="knn-k">K = {knnK}</Label>
                  </div>
                  <Slider
                    min={1} max={20} step={1}
                    value={[knnK]}
                    onValueChange={(value) => setKnnK(value[0])}
                  />
                  <Button onClick={handleGenerateKNNPlot} disabled={isGeneratingPlot} className="w-full">
                    {isGeneratingPlot ? "Generating..." : "Optimize K"}
                  </Button>
                  {optimizationPlot && <img src={optimizationPlot} className="w-full rounded-md border" />}
                </div>
              )}

              <Button onClick={handleFillMissingValues} disabled={isProcessing || selectedColumnsForMissing.length === 0} className="w-full">
                {isProcessing ? "Processing..." : "Apply Fill"}
              </Button>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="normalize">
          <Card>
            <CardHeader><CardTitle>Normalize Data</CardTitle></CardHeader>
            <CardContent className="space-y-4">
              <div className="space-y-2">
                <Label>Select Columns</Label>
                <div className="grid grid-cols-3 gap-2 border rounded-md p-3 max-h-40 overflow-y-auto">
                  {columns.map((column) => (
                    <div key={column} className="flex items-center space-x-2">
                      <Checkbox
                        id={`normalize-${column}`}
                        checked={selectedColumnsForNormalization.includes(column)}
                        onCheckedChange={(checked) => {
                          if (checked) setSelectedColumnsForNormalization([...selectedColumnsForNormalization, column]);
                          else setSelectedColumnsForNormalization(selectedColumnsForNormalization.filter(col => col !== column));
                        }}
                      />
                      <Label htmlFor={`normalize-${column}`}>{column}</Label>
                    </div>
                  ))}
                </div>
              </div>
              <Select value={normalizationMethod} onValueChange={setNormalizationMethod}>
                <SelectTrigger><SelectValue placeholder="Method" /></SelectTrigger>
                <SelectContent>
                  <SelectItem value="zscore">Z-Score</SelectItem>
                  <SelectItem value="minmax">Min-Max</SelectItem>
                </SelectContent>
              </Select>
              <Button onClick={handleNormalizeData} disabled={isProcessing || selectedColumnsForNormalization.length === 0} className="w-full">
                Apply Normalization
              </Button>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="categorical">
          <Card>
            <CardHeader><CardTitle>Categorical to Numerical</CardTitle></CardHeader>
            <CardContent className="space-y-4">
              <div className="space-y-2">
                <Label>Select Columns</Label>
                <div className="grid grid-cols-3 gap-2 border rounded-md p-3 max-h-40 overflow-y-auto">
                  {columns.map((column) => (
                    <div key={column} className="flex items-center space-x-2">
                      <Checkbox
                        id={`categorical-${column}`}
                        checked={selectedColumnsForCategorical.includes(column)}
                        onCheckedChange={(checked) => {
                          if (checked) setSelectedColumnsForCategorical([...selectedColumnsForCategorical, column]);
                          else setSelectedColumnsForCategorical(selectedColumnsForCategorical.filter(col => col !== column));
                        }}
                      />
                      <Label htmlFor={`categorical-${column}`}>{column}</Label>
                    </div>
                  ))}
                </div>
              </div>
              <RadioGroup value={encodingMethod} onValueChange={setEncodingMethod}>
                <div className="flex items-center space-x-2">
                  <RadioGroupItem value="label" id="label" /><Label htmlFor="label">Label Encoding</Label>
                </div>
                <div className="flex items-center space-x-2">
                  <RadioGroupItem value="onehot" id="onehot" /><Label htmlFor="onehot">One-Hot Encoding</Label>
                </div>
              </RadioGroup>
              <Button onClick={handleCategoricalToNumerical} disabled={isProcessing || selectedColumnsForCategorical.length === 0} className="w-full">
                Convert
              </Button>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="advanced">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <Card>
              <CardHeader><CardTitle>Binarization</CardTitle></CardHeader>
              <CardContent className="space-y-4">
                <Select value={selectedColumnForBinarize} onValueChange={setSelectedColumnForBinarize}>
                  <SelectTrigger><SelectValue placeholder="Select column" /></SelectTrigger>
                  <SelectContent>{columns.map(c => <SelectItem key={c} value={c}>{c}</SelectItem>)}</SelectContent>
                </Select>
                <div className="space-y-1">
                  <Label className="text-xs">Zero Group (comma separated)</Label>
                  <input className="flex h-9 w-full rounded-md border border-input px-3 py-1 text-sm" placeholder="e.g. CL0, CL1" value={binarizeZeroGroup} onChange={(e) => setBinarizeZeroGroup(e.target.value)} />
                </div>
                <div className="space-y-1">
                  <Label className="text-xs">OR Threshold</Label>
                  <input type="number" className="flex h-9 w-full rounded-md border border-input px-3 py-1 text-sm" placeholder="e.g. 0.5" value={binarizeThreshold} onChange={(e) => setBinarizeThreshold(e.target.value)} />
                </div>
                <Button onClick={handleBinarize} disabled={isProcessing || !selectedColumnForBinarize} className="w-full">Apply</Button>
              </CardContent>
            </Card>

            <Card>
              <CardHeader><CardTitle>Ordinal Mapping</CardTitle></CardHeader>
              <CardContent className="space-y-4">
                <Select value={selectedColumnForOrdinal} onValueChange={setSelectedColumnForOrdinal}>
                  <SelectTrigger><SelectValue placeholder="Select column" /></SelectTrigger>
                  <SelectContent>{columns.map(c => <SelectItem key={c} value={c}>{c}</SelectItem>)}</SelectContent>
                </Select>
                <div className="space-y-1">
                  <Label className="text-xs">Mapping Order (comma separated)</Label>
                  <textarea className="flex min-h-[60px] w-full rounded-md border border-input px-3 py-1 text-sm" placeholder="e.g. Young, Middle, Old" value={ordinalOrder} onChange={(e) => setOrdinalOrder(e.target.value)} />
                </div>
                <Button onClick={handleOrdinalMap} disabled={isProcessing || !selectedColumnForOrdinal} className="w-full">Apply</Button>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        <TabsContent value="save">
          <Card>
            <CardHeader><CardTitle>Save Data</CardTitle></CardHeader>
            <CardContent className="space-y-4">
              <div className="flex gap-2">
                <input className="flex h-10 w-full rounded-md border border-input px-3 py-2 text-sm" placeholder="filename.csv" value={savePath} onChange={(e) => setSavePath(e.target.value)} />
                <Button onClick={handleSaveData} disabled={isProcessing || !savePath}><Save className="mr-2 h-4 w-4" />Save</Button>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
      {error && <div className="bg-destructive/10 text-destructive p-3 rounded-md text-sm">{error}</div>}
    </div>
  );
}
