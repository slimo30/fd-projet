"use client";

import { useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Settings, PlayCircle, BarChart3, AlertCircle } from "lucide-react";
import { useData } from "@/context/DataContext";
import { Alert, AlertDescription } from "@/components/ui/alert";
import AlgorithmSelection from "./components/algorithm-selection";
import ModelConfiguration from "./components/model-configuration";
import ModelResults from "./components/model-results";
import ComparisonView from "./components/comparison-view";


export default function MLPage() {
  const { dataPath, columns, hasData } = useData();
  const [selectedAlgorithm, setSelectedAlgorithm] = useState<string>("");
  const [selectedAlgorithmName, setSelectedAlgorithmName] = useState<string>("");
  const [results, setResults] = useState<any>(null);
  const [activeTab, setActiveTab] = useState<string>("algorithm");

  const handleAlgorithmSelected = (algorithm: string) => {
    setSelectedAlgorithm(algorithm);
    // Assuming algorithm name can be derived or is the same as the id for now
    const algorithmName = algorithm.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
    setSelectedAlgorithmName(algorithmName);
    setActiveTab("configuration");
  };

  const handleModelRun = (modelResults: any) => {
    setResults(modelResults);
    setActiveTab("results");
  };

  if (!hasData) {
    return (
      <div className="container mx-auto py-6 px-4 max-w-7xl">
        <div className="mb-8">
          <h1 className="text-3xl font-bold tracking-tight">Machine Learning</h1>
          <p className="text-muted-foreground mt-1">
            Train and evaluate machine learning models
          </p>
        </div>

        <Alert>
          <AlertCircle className="h-4 w-4" />
          <AlertDescription>
            Please upload data in the Data Processing page first
          </AlertDescription>
        </Alert>
      </div>
    );
  }

  return (
    <div className="container mx-auto py-6 px-4 max-w-7xl">
      <div className="mb-8">
        <h1 className="text-3xl font-bold tracking-tight">Machine Learning</h1>
        <p className="text-muted-foreground mt-1">
          Train and evaluate machine learning models
        </p>
      </div>

      <Tabs value={activeTab} onValueChange={setActiveTab}>
        <TabsList className="grid w-full grid-cols-4 mb-6">
          <TabsTrigger value="algorithm" className="gap-2">
            <Settings className="h-4 w-4" />
            <span className="hidden sm:inline">Algorithm</span>
          </TabsTrigger>
          <TabsTrigger value="configuration" disabled={!selectedAlgorithm} className="gap-2">
            <PlayCircle className="h-4 w-4" />
            <span className="hidden sm:inline">Configure</span>
          </TabsTrigger>
          <TabsTrigger value="results" disabled={!results} className="gap-2">
            <BarChart3 className="h-4 w-4" />
            <span className="hidden sm:inline">Results</span>
          </TabsTrigger>

          <TabsTrigger value="comparison" className="gap-2">
            <BarChart3 className="h-4 w-4" />
            <span className="hidden sm:inline">Compare</span>
          </TabsTrigger>
        </TabsList>

        <TabsContent value="algorithm">
          <Card>
            <CardHeader>
              <CardTitle>Select Algorithm</CardTitle>
              <CardDescription>
                Choose a machine learning algorithm
              </CardDescription>
            </CardHeader>
            <CardContent>
              <AlgorithmSelection
                onAlgorithmSelected={handleAlgorithmSelected}
              />
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="configuration">
          {selectedAlgorithm && (
            <Card>
              <CardHeader>
                <CardTitle>Configure & Run</CardTitle>
                <CardDescription>
                  Set parameters for {selectedAlgorithmName || selectedAlgorithm}
                </CardDescription>
              </CardHeader>
              <CardContent>
                <ModelConfiguration
                  dataPath={dataPath!}
                  columns={columns}
                  algorithm={selectedAlgorithm}
                  onModelRun={handleModelRun}
                />
              </CardContent>
            </Card>
          )}
        </TabsContent>

        <TabsContent value="results">
          {results && (
            <Card>
              <CardHeader>
                <CardTitle>Results</CardTitle>
                <CardDescription>
                  Performance metrics
                </CardDescription>
              </CardHeader>
              <CardContent>
                <ModelResults results={results} algorithm={selectedAlgorithm} />
              </CardContent>
            </Card>
          )}
        </TabsContent>



        <TabsContent value="comparison">
          <Card>
            <CardHeader>
              <CardTitle>Compare Algorithms</CardTitle>
              <CardDescription>
                Performance comparison
              </CardDescription>
            </CardHeader>
            <CardContent>
              <ComparisonView />
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}
