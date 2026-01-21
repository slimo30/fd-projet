"use client";

import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Brain, Network, TreeDeciduous, TrendingUp, Layers } from "lucide-react";

interface AlgorithmSelectionProps {
  onAlgorithmSelected: (algorithm: string, algorithmName: string) => void;
}

const algorithms = [
  {
    id: "knn",
    name: "K-Nearest Neighbors (K-NN)",
    description: "Classification algorithm that finds the K nearest data points",
    icon: Network,
    color: "text-blue-500",
  },
  {
    id: "naive-bayes",
    name: "Naive Bayes",
    description: "Probabilistic classifier based on Bayes' theorem",
    icon: Brain,
    color: "text-purple-500",
  },
  {
    id: "decision-tree",
    name: "Decision Tree",
    description: "Tree-based classification using ID3, C4.5, or CART",
    icon: TreeDeciduous,
    color: "text-green-500",
  },
  {
    id: "linear-regression",
    name: "Linear Regression",
    description: "Predict continuous values using linear relationships",
    icon: TrendingUp,
    color: "text-orange-500",
  },
  {
    id: "neural-network",
    name: "Neural Network",
    description: "Multi-layer perceptron for complex pattern recognition",
    icon: Layers,
    color: "text-red-500",
  },
];

export default function AlgorithmSelection({ onAlgorithmSelected }: AlgorithmSelectionProps) {
  const [selected, setSelected] = useState<string>("");

  const handleSelect = (algorithmId: string, algorithmName: string) => {
    setSelected(algorithmId);
    onAlgorithmSelected(algorithmId, algorithmName);
  };

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
      {algorithms.map((algo) => {
        const Icon = algo.icon;
        return (
          <Card
            key={algo.id}
            className={`cursor-pointer transition-all hover:shadow-lg ${
              selected === algo.id ? "ring-2 ring-primary" : ""
            }`}
            onClick={() => handleSelect(algo.id, algo.name)}
          >
            <CardHeader>
              <div className="flex items-center gap-3">
                <Icon className={`h-6 w-6 ${algo.color}`} />
                <CardTitle className="text-lg">{algo.name}</CardTitle>
              </div>
            </CardHeader>
            <CardContent>
              <CardDescription>{algo.description}</CardDescription>
            </CardContent>
          </Card>
        );
      })}
    </div>
  );
}
