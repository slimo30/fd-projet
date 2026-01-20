"use client"

import { useState } from "react"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Alert, AlertDescription } from "@/components/ui/alert"
import { AlertCircle, Settings, TrendingUp, GitBranch, PlayCircle, Zap, BarChart3 } from "lucide-react"
import { useData } from "@/context/DataContext"
import AlgorithmSelection from "./components/algorithm-selection"
import ElbowMethod from "./components/elbow-method"
import Dendrogram from "./components/dendrogram"
import RunClustering from "./components/run-clustering"
import RunDBSCAN from "./components/run-dbscan"
import ComparisonView from "./components/comparison-view"
import { Toaster } from "sonner"

export default function ClusteringPage() {
  const { dataPath, hasData } = useData()
  const [selectedAlgorithm, setSelectedAlgorithm] = useState<string>("kmeans")

  if (!hasData) {
    return (
      <div className="container mx-auto py-6 px-4 max-w-7xl">
        <div className="mb-8">
          <h1 className="text-3xl font-bold tracking-tight">Clustering Analysis</h1>
          <p className="text-muted-foreground mt-1">
            Discover patterns in your data
          </p>
        </div>
        
        <Alert>
          <AlertCircle className="h-4 w-4" />
          <AlertDescription>
            Please upload data in the Data Processing page first
          </AlertDescription>
        </Alert>
      </div>
    )
  }

  return (
    <div className="container mx-auto py-6 px-4 max-w-7xl">
      <div className="mb-8">
        <h1 className="text-3xl font-bold tracking-tight">Clustering Analysis</h1>
        <p className="text-muted-foreground mt-1">
          Discover patterns in your data
        </p>
      </div>

      <Tabs defaultValue="algorithm" className="w-full">
        <TabsList className="grid grid-cols-6 mb-6">
          <TabsTrigger value="algorithm" className="gap-2">
            <Settings className="h-4 w-4" />
            <span className="hidden sm:inline">Algorithm</span>
          </TabsTrigger>
          <TabsTrigger value="elbow" className="gap-2">
            <TrendingUp className="h-4 w-4" />
            <span className="hidden sm:inline">Elbow</span>
          </TabsTrigger>
          <TabsTrigger value="dendrogram" className="gap-2">
            <GitBranch className="h-4 w-4" />
            <span className="hidden sm:inline">Dendrogram</span>
          </TabsTrigger>
          <TabsTrigger value="run" className="gap-2">
            <PlayCircle className="h-4 w-4" />
            <span className="hidden sm:inline">Run</span>
          </TabsTrigger>
          <TabsTrigger value="dbscan" className="gap-2">
            <Zap className="h-4 w-4" />
            <span className="hidden sm:inline">DBSCAN</span>
          </TabsTrigger>
          <TabsTrigger value="comparison" className="gap-2">
            <BarChart3 className="h-4 w-4" />
            <span className="hidden sm:inline">Compare</span>
          </TabsTrigger>
        </TabsList>

        <TabsContent value="algorithm">
          <Card>
            <CardHeader>
              <CardTitle>Algorithm Selection</CardTitle>
              <CardDescription>Select the clustering algorithm to use</CardDescription>
            </CardHeader>
            <CardContent>
              <AlgorithmSelection filePath={dataPath!} onAlgorithmSelect={setSelectedAlgorithm} />
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="elbow">
          <Card>
            <CardHeader>
              <CardTitle>Elbow Method</CardTitle>
              <CardDescription>Determine the optimal number of clusters</CardDescription>
            </CardHeader>
            <CardContent>
              <ElbowMethod filePath={dataPath!} algorithm={selectedAlgorithm} />
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="dendrogram">
          <Card>
            <CardHeader>
              <CardTitle>Dendrogram</CardTitle>
              <CardDescription>Hierarchical clustering visualization</CardDescription>
            </CardHeader>
            <CardContent>
              <Dendrogram filePath={dataPath!} />
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="run">
          <Card>
            <CardHeader>
              <CardTitle>Run Clustering</CardTitle>
              <CardDescription>Execute clustering algorithm and view results</CardDescription>
            </CardHeader>
            <CardContent>
              <RunClustering filePath={dataPath!} algorithm={selectedAlgorithm} />
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="dbscan">
          <Card>
            <CardHeader>
              <CardTitle>DBSCAN Clustering</CardTitle>
              <CardDescription>Density-based spatial clustering</CardDescription>
            </CardHeader>
            <CardContent>
              <RunDBSCAN filePath={dataPath!} />
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="comparison">
          <Card>
            <CardHeader>
              <CardTitle>Algorithm Comparison</CardTitle>
              <CardDescription>Compare performance metrics across algorithms</CardDescription>
            </CardHeader>
            <CardContent>
              <ComparisonView />
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
      <Toaster position="bottom-right" />
    </div>
  )
}
