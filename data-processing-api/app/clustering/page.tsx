"use client"

import { useState } from "react"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import UploadData from "./components/upload-data"
import AlgorithmSelection from "./components/algorithm-selection"
import ElbowMethod from "./components/elbow-method"
import Dendrogram from "./components/dendrogram"
import RunClustering from "./components/run-clustering"
import RunDBSCAN from "./components/run-dbscan"
import ComparisonView from "./components/comparison-view"
import { Toaster, toast } from "sonner"

export default function ClusteringPage() {
  const [uploadedFilePath, setUploadedFilePath] = useState<string | null>(null)
  const [selectedAlgorithm, setSelectedAlgorithm] = useState<string>("kmeans")

  const handleUploadSuccess = (path: string) => {
    setUploadedFilePath(path)
    toast.success("Data uploaded successfully", {
      description: "Your data is ready for clustering analysis.",
    })
  }

  const handleAlgorithmSelect = (algorithm: string) => {
    setSelectedAlgorithm(algorithm)
    toast.success("Algorithm selected", {
      description: `Selected ${algorithm.toUpperCase()} for clustering.`,
    })
  }

  return (
    <div className="container mx-auto py-8">
      <h1 className="text-3xl font-bold mb-6">Clustering Analysis Dashboard</h1>

      <div className="grid grid-cols-1 gap-6">
        <Card>
          <CardHeader>
            <CardTitle>Data Upload</CardTitle>
            <CardDescription>Upload your CSV data file for clustering analysis</CardDescription>
          </CardHeader>
          <CardContent>
            <UploadData onUploadSuccess={handleUploadSuccess} />
          </CardContent>
        </Card>

        {uploadedFilePath && (
          <Tabs defaultValue="algorithm" className="w-full">
            <TabsList className="grid grid-cols-6 mb-4">
              <TabsTrigger value="algorithm">Algorithm</TabsTrigger>
              <TabsTrigger value="elbow">Elbow Method</TabsTrigger>
              <TabsTrigger value="dendrogram">Dendrogram</TabsTrigger>
              <TabsTrigger value="run">Run Clustering</TabsTrigger>
              <TabsTrigger value="dbscan">DBSCAN</TabsTrigger>
              <TabsTrigger value="comparison">Comparison</TabsTrigger>
            </TabsList>

            <TabsContent value="algorithm">
              <Card>
                <CardHeader>
                  <CardTitle>Algorithm Selection</CardTitle>
                  <CardDescription>Select the clustering algorithm to use</CardDescription>
                </CardHeader>
                <CardContent>
                  <AlgorithmSelection filePath={uploadedFilePath} onAlgorithmSelect={handleAlgorithmSelect} />
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
                  <ElbowMethod filePath={uploadedFilePath} algorithm={selectedAlgorithm} />
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
                  <Dendrogram filePath={uploadedFilePath} />
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
                  <RunClustering filePath={uploadedFilePath} algorithm={selectedAlgorithm} />
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
                  <RunDBSCAN filePath={uploadedFilePath} />
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
        )}
      </div>
      <Toaster position="bottom-right" />
    </div>
  )
}
