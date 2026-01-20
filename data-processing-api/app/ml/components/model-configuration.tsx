"use client";

import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Loader2, AlertCircle } from "lucide-react";
import { toast } from "sonner";
import { mlApi, ApiError } from "@/lib/api";

interface ModelConfigurationProps {
  dataPath: string;
  columns: string[];
  algorithm: string;
  onModelRun: (results: any) => void;
}

export default function ModelConfiguration({
  dataPath,
  columns,
  algorithm,
  onModelRun,
}: ModelConfigurationProps) {
  const [targetColumn, setTargetColumn] = useState<string>("");
  const [isRunning, setIsRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // K-NN specific
  const [maxK, setMaxK] = useState<number>(10);

  // Decision Tree specific
  const [treeAlgorithm, setTreeAlgorithm] = useState<string>("cart");

  // Neural Network specific
  const [hiddenLayers, setHiddenLayers] = useState<string>("100");

  const handleRunModel = async () => {
    if (!targetColumn) {
      setError("Please select a target column");
      return;
    }

    setIsRunning(true);
    setError(null);

    try {
      let result;

      switch (algorithm) {
        case "knn":
          result = await mlApi.runKNN(dataPath, targetColumn, maxK);
          break;

        case "naive-bayes":
          result = await mlApi.runNaiveBayes(dataPath, targetColumn);
          break;

        case "decision-tree":
          result = await mlApi.runDecisionTree(
            dataPath,
            targetColumn,
            treeAlgorithm as 'id3' | 'c4.5' | 'cart'
          );
          break;

        case "linear-regression":
          result = await mlApi.runLinearRegression(dataPath, targetColumn);
          break;

        case "neural-network":
          const layers = hiddenLayers.split(",").map(l => parseInt(l.trim())).filter(l => !isNaN(l));
          result = await mlApi.runNeuralNetwork(dataPath, targetColumn, layers);
          break;

        default:
          throw new Error("Unknown algorithm");
      }

      onModelRun(result);
      toast.success("Model trained successfully!");
    } catch (err: any) {
      const errorMessage = err instanceof ApiError ? err.message : (err.message || "An error occurred");
      setError(errorMessage);
      toast.error("Model training failed", {
        description: errorMessage,
      });
    } finally {
      setIsRunning(false);
    }
  };

  return (
    <div className="space-y-6">
      <div className="space-y-2">
        <Label htmlFor="target">Target Column</Label>
        <Select value={targetColumn} onValueChange={setTargetColumn}>
          <SelectTrigger id="target">
            <SelectValue placeholder="Select target column" />
          </SelectTrigger>
          <SelectContent>
            {columns.map((col) => (
              <SelectItem key={col} value={col}>
                {col}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
        <p className="text-sm text-muted-foreground">
          The column you want to predict
        </p>
      </div>

      {algorithm === "knn" && (
        <div className="space-y-2">
          <Label htmlFor="maxK">Maximum K Value</Label>
          <Input
            id="maxK"
            type="number"
            min={1}
            max={50}
            value={maxK}
            onChange={(e) => setMaxK(parseInt(e.target.value) || 10)}
          />
          <p className="text-sm text-muted-foreground">
            The algorithm will test K values from 1 to {maxK} and find the best one
          </p>
        </div>
      )}

      {algorithm === "decision-tree" && (
        <div className="space-y-2">
          <Label htmlFor="treeAlgo">Tree Algorithm</Label>
          <Select value={treeAlgorithm} onValueChange={setTreeAlgorithm}>
            <SelectTrigger id="treeAlgo">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="id3">ID3 (Iterative Dichotomiser 3)</SelectItem>
              <SelectItem value="c4.5">C4.5 (Successor of ID3)</SelectItem>
              <SelectItem value="cart">CART (Classification and Regression Trees)</SelectItem>
            </SelectContent>
          </Select>
          <p className="text-sm text-muted-foreground">
            Choose the decision tree splitting criterion
          </p>
        </div>
      )}

      {algorithm === "neural-network" && (
        <div className="space-y-2">
          <Label htmlFor="layers">Hidden Layers</Label>
          <Input
            id="layers"
            type="text"
            value={hiddenLayers}
            onChange={(e) => setHiddenLayers(e.target.value)}
            placeholder="100, 50"
          />
          <p className="text-sm text-muted-foreground">
            Comma-separated list of neurons per hidden layer (e.g., "100" or "100, 50")
          </p>
        </div>
      )}

      {error && (
        <Alert variant="destructive">
          <AlertCircle className="h-4 w-4" />
          <AlertTitle>Error</AlertTitle>
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}

      <Button
        onClick={handleRunModel}
        disabled={isRunning || !targetColumn}
        className="w-full"
        size="lg"
      >
        {isRunning ? (
          <>
            <Loader2 className="mr-2 h-4 w-4 animate-spin" />
            Training Model...
          </>
        ) : (
          "Run Model"
        )}
      </Button>
    </div>
  );
}
