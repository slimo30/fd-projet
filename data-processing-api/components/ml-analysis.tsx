"use client";

import { useState } from "react";
import { Loader2, Play } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { toast } from "sonner";
import Image from "next/image";

interface MlAnalysisProps {
  path: string;
  columns: string[];
}

export default function MlAnalysis({ path, columns }: MlAnalysisProps) {
  const [algorithm, setAlgorithm] = useState<string>("knn");
  const [targetColumn, setTargetColumn] = useState<string>("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<any>(null);

  // Algorithm specific parameters
  const [knnK, setKnnK] = useState<string>("20");
  const [dtAlgorithm, setDtAlgorithm] = useState<string>("cart");
  const [nnHiddenLayers, setNnHiddenLayers] = useState<string>("(100,)");

  const runAnalysis = async () => {
    if (!targetColumn) {
      toast.error("Please select a target column");
      return;
    }

    setLoading(true);
    setResult(null);

    let endpoint = "";
    let payload: any = { path, target: targetColumn };

    switch (algorithm) {
      case "knn":
        endpoint = "/api/ml/knn";
        payload.max_k = parseInt(knnK);
        break;
      case "naive-bayes":
        endpoint = "/api/ml/naive-bayes";
        break;
      case "decision-tree":
        endpoint = "/api/ml/decision-tree";
        payload.algorithm_type = dtAlgorithm;
        break;
      case "linear-regression":
        endpoint = "/api/ml/linear-regression";
        break;
      case "neural-network":
        endpoint = "/api/ml/neural-network";
        payload.hidden_layers = nnHiddenLayers;
        break;
    }

    try {
      const response = await fetch(endpoint, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || "Analysis failed");
      }

      setResult(data);
      toast.success("Analysis completed successfully");
    } catch (error) {
      console.error("ML Error:", error);
      toast.error(error instanceof Error ? error.message : "Analysis failed");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {/* Configuration Panel */}
        <Card className="md:col-span-1">
          <CardHeader>
            <CardTitle>Configuration</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-2">
              <Label>Algorithm</Label>
              <Select value={algorithm} onValueChange={setAlgorithm}>
                <SelectTrigger>
                  <SelectValue placeholder="Select algorithm" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="knn">K-Nearest Neighbors (K-NN)</SelectItem>
                  <SelectItem value="naive-bayes">Naive Bayes</SelectItem>
                  <SelectItem value="decision-tree">Decision Tree</SelectItem>
                  <SelectItem value="linear-regression">
                    Linear Regression
                  </SelectItem>
                  <SelectItem value="neural-network">Neural Network</SelectItem>
                </SelectContent>
              </Select>
            </div>

            <div className="space-y-2">
              <Label>Target Column</Label>
              <Select value={targetColumn} onValueChange={setTargetColumn}>
                <SelectTrigger>
                  <SelectValue placeholder="Select target variable" />
                </SelectTrigger>
                <SelectContent>
                  {columns.map((col) => (
                    <SelectItem key={col} value={col}>
                      {col}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            {/* Dynamic Parameters */}
            {algorithm === "knn" && (
              <div className="space-y-2">
                <Label>Max K (for optimization)</Label>
                <Input
                  type="number"
                  value={knnK}
                  onChange={(e) => setKnnK(e.target.value)}
                  min="1"
                />
              </div>
            )}

            {algorithm === "decision-tree" && (
              <div className="space-y-2">
                <Label>Tree Algorithm</Label>
                <Select value={dtAlgorithm} onValueChange={setDtAlgorithm}>
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="cart">CART (Gini)</SelectItem>
                    <SelectItem value="id3">ID3 (Entropy)</SelectItem>
                    <SelectItem value="c4.5">C4.5 (Entropy)</SelectItem>
                  </SelectContent>
                </Select>
              </div>
            )}

            {algorithm === "neural-network" && (
              <div className="space-y-2">
                <Label>Hidden Layers</Label>
                <Input
                  value={nnHiddenLayers}
                  onChange={(e) => setNnHiddenLayers(e.target.value)}
                  placeholder="(100,) or (10, 10)"
                />
                <p className="text-xs text-muted-foreground">
                  Tuple format e.g. (100,) for one layer
                </p>
              </div>
            )}

            <Button
              className="w-full mt-4"
              onClick={runAnalysis}
              disabled={loading}
            >
              {loading ? (
                <>
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  Running...
                </>
              ) : (
                <>
                  <Play className="mr-2 h-4 w-4" />
                  Run Analysis
                </>
              )}
            </Button>
          </CardContent>
        </Card>

        {/* Results Panel */}
        <div className="md:col-span-2 space-y-6">
          {result && (
            <>
              {/* Metrics Card */}
              <Card>
                <CardHeader>
                  <CardTitle>Performance Metrics</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
                    {Object.entries(result.metrics).map(([key, value]) => {
                       if (key.includes('url') || Array.isArray(value)) return null;
                       return (
                        <div key={key} className="p-3 bg-muted rounded-lg">
                          <p className="text-xs font-medium text-muted-foreground capitalize">
                            {key.replace(/_/g, " ")}
                          </p>
                          <p className="text-xl font-bold">
                            {typeof value === "number"
                              ? value.toFixed(4)
                              : String(value)}
                          </p>
                        </div>
                      );
                    })}
                  </div>
                </CardContent>
              </Card>

              {/* Visualization Card */}
              {(result.plot_url || (result.metrics && result.metrics.matrix_plot_url)) && (
                <Card>
                  <CardHeader>
                    <CardTitle>Visualization</CardTitle>
                  </CardHeader>
                  <CardContent className="flex justify-center bg-white rounded-b-lg p-4">
                    <div className="relative w-full h-[400px]">
                      <Image
                        src={result.metrics?.matrix_plot_url || result.plot_url}
                        alt="Model Visualization"
                        fill
                        className="object-contain"
                        unoptimized // Since we serve from Flask static
                      />
                    </div>
                  </CardContent>
                </Card>
              )}
            </>
          )}

          {!result && !loading && (
            <div className="h-full flex items-center justify-center border-2 border-dashed rounded-lg p-12 text-muted-foreground">
              Select an algorithm and run analysis to see results
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
