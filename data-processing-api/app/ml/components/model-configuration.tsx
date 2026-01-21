"use client";

import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Loader2, AlertCircle, Plus, X, Layers } from "lucide-react";
import { toast } from "sonner";
import { mlApi, ApiError } from "@/lib/api";
import { Badge } from "@/components/ui/badge";

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

  // Neural Network specific - now using array
  const [neuralLayers, setNeuralLayers] = useState<number[]>([100]);

  // Preset configurations for neural networks
  const neuralPresets = [
    { name: "Simple (1 layer)", layers: [100], description: "Fast training, basic patterns" },
    { name: "Medium (2 layers)", layers: [100, 50], description: "Balanced performance" },
    { name: "Deep (3 layers)", layers: [128, 64, 32], description: "Complex patterns" },
    { name: "Wide (2 layers)", layers: [200, 100], description: "Large capacity" },
  ];

  const addNeuralLayer = () => {
    setNeuralLayers([...neuralLayers, 50]);
  };

  const removeNeuralLayer = (index: number) => {
    if (neuralLayers.length > 1) {
      setNeuralLayers(neuralLayers.filter((_, i) => i !== index));
    }
  };

  const updateNeuralLayer = (index: number, value: number) => {
    const newLayers = [...neuralLayers];
    newLayers[index] = Math.max(1, Math.min(1000, value)); // Clamp between 1 and 1000
    setNeuralLayers(newLayers);
  };

  const applyPreset = (layers: number[]) => {
    setNeuralLayers(layers);
  };

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
          result = await mlApi.runNeuralNetwork(dataPath, targetColumn, neuralLayers);
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
              <SelectItem value="chaid">CHAID (Chi-square Automatic Interaction Detection)</SelectItem>
            </SelectContent>
          </Select>
          <p className="text-sm text-muted-foreground">
            Choose the decision tree splitting criterion
          </p>
        </div>
      )}

      {algorithm === "neural-network" && (
        <div className="space-y-4">
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <Label className="flex items-center gap-2">
                <Layers className="h-4 w-4" />
                Network Architecture
              </Label>
              <Badge variant="secondary" className="font-mono">
                {neuralLayers.length} {neuralLayers.length === 1 ? 'Layer' : 'Layers'}
              </Badge>
            </div>
            
            <p className="text-sm text-muted-foreground">
              Configure the number of neurons in each hidden layer
            </p>

            {/* Preset Buttons */}
            <div className="grid grid-cols-2 gap-2">
              {neuralPresets.map((preset, idx) => (
                <Button
                  key={idx}
                  variant={JSON.stringify(neuralLayers) === JSON.stringify(preset.layers) ? "default" : "outline"}
                  size="sm"
                  onClick={() => applyPreset(preset.layers)}
                  className="flex flex-col items-start h-auto py-2 px-3"
                >
                  <span className="font-semibold text-xs">{preset.name}</span>
                  <span className="text-xs opacity-70 font-normal">{preset.description}</span>
                  <span className="text-xs font-mono mt-1">[{preset.layers.join(', ')}]</span>
                </Button>
              ))}
            </div>

            {/* Custom Layer Configuration */}
            <div className="border rounded-lg p-4 space-y-3 bg-gray-50/50">
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium">Custom Configuration</span>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={addNeuralLayer}
                  disabled={neuralLayers.length >= 5}
                  className="h-8 gap-1"
                >
                  <Plus className="h-3 w-3" />
                  Add Layer
                </Button>
              </div>

              <div className="space-y-2">
                {neuralLayers.map((neurons, index) => (
                  <div key={index} className="flex items-center gap-2">
                    <Badge variant="outline" className="min-w-[80px] justify-center">
                      Layer {index + 1}
                    </Badge>
                    <Input
                      type="number"
                      min={1}
                      max={1000}
                      value={neurons}
                      onChange={(e) => updateNeuralLayer(index, parseInt(e.target.value) || 1)}
                      className="flex-1"
                      placeholder="Neurons"
                    />
                    <div className="flex items-center gap-1 min-w-[80px]">
                      <span className="text-xs text-muted-foreground">{neurons} neurons</span>
                    </div>
                    {neuralLayers.length > 1 && (
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={() => removeNeuralLayer(index)}
                        className="h-8 w-8 p-0"
                      >
                        <X className="h-4 w-4" />
                      </Button>
                    )}
                  </div>
                ))}
              </div>

              {neuralLayers.length >= 5 && (
                <p className="text-xs text-amber-600">
                  Maximum of 5 layers reached
                </p>
              )}
            </div>

            {/* Visual representation */}
            <div className="border rounded-lg p-3 bg-white">
              <div className="text-xs font-medium mb-2 text-muted-foreground">Network Visualization</div>
              <div className="flex items-center gap-2 overflow-x-auto pb-2">
                <div className="flex flex-col items-center min-w-[60px]">
                  <div className="w-12 h-12 rounded-lg bg-blue-100 flex items-center justify-center border-2 border-blue-300">
                    <span className="text-xs font-semibold text-blue-700">Input</span>
                  </div>
                  <span className="text-xs text-muted-foreground mt-1">Features</span>
                </div>
                
                {neuralLayers.map((neurons, idx) => (
                  <div key={idx} className="flex items-center gap-2">
                    <div className="text-muted-foreground">→</div>
                    <div className="flex flex-col items-center min-w-[60px]">
                      <div 
                        className="w-12 h-12 rounded-lg bg-purple-100 flex items-center justify-center border-2 border-purple-300"
                        style={{ 
                          opacity: 1 - (idx * 0.15),
                        }}
                      >
                        <span className="text-xs font-bold text-purple-700">{neurons}</span>
                      </div>
                      <span className="text-xs text-muted-foreground mt-1">Layer {idx + 1}</span>
                    </div>
                  </div>
                ))}
                
                <div className="flex items-center gap-2">
                  <div className="text-muted-foreground">→</div>
                  <div className="flex flex-col items-center min-w-[60px]">
                    <div className="w-12 h-12 rounded-lg bg-green-100 flex items-center justify-center border-2 border-green-300">
                      <span className="text-xs font-semibold text-green-700">Output</span>
                    </div>
                    <span className="text-xs text-muted-foreground mt-1">Prediction</span>
                  </div>
                </div>
              </div>
            </div>

            <div className="text-xs text-muted-foreground bg-blue-50 border border-blue-200 rounded p-2">
              <strong>Tip:</strong> More layers = deeper network (better for complex patterns but slower training). 
              More neurons = wider network (more capacity but risk of overfitting).
            </div>
          </div>
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
