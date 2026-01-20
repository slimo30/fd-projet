"use client"

import { useState } from "react"
import { Upload, FileUp, Database, BarChart4, Table, Save } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import FileUploader from "@/components/file-uploader"
import DataPreview from "@/components/data-preview"
import DataStatistics from "@/components/data-statistics"
import DataProcessing from "@/components/data-processing"
import DataVisualization from "@/components/data-visualization"
import MlAnalysis from "@/components/ml-analysis"
import MlComparison from "@/components/ml-comparison"

export default function Home() {
  const [currentPath, setCurrentPath] = useState<string | null>(null)
  const [columns, setColumns] = useState<string[]>([])
  const [activeTab, setActiveTab] = useState("upload")

  const handleFileUploaded = async (path: string) => {
    setCurrentPath(path)

    // Fetch columns
    try {
      const response = await fetch(`/api/data/columns?path=${path}`)
      const data = await response.json()
      if (data.columns) {
        setColumns(data.columns)
      }
    } catch (error) {
      console.error("Error fetching columns:", error)
    }

    // Move to the next tab
    setActiveTab("preview")
  }

  return (
    <main className="container mx-auto py-6 px-4 md:px-6 lg:px-8">
      <h1 className="text-3xl font-bold mb-6">Data Processing Platform</h1>

      <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
        <TabsList className="grid grid-cols-5 mb-8">
          <TabsTrigger value="upload" className="flex items-center gap-2">
            <Upload className="h-4 w-4" />
            <span className="hidden sm:inline">Upload</span>
          </TabsTrigger>
          <TabsTrigger value="preview" disabled={!currentPath} className="flex items-center gap-2">
            <Table className="h-4 w-4" />
            <span className="hidden sm:inline">Preview</span>
          </TabsTrigger>
          <TabsTrigger value="statistics" disabled={!currentPath} className="flex items-center gap-2">
            <Database className="h-4 w-4" />
            <span className="hidden sm:inline">Statistics</span>
          </TabsTrigger>
          <TabsTrigger value="processing" disabled={!currentPath} className="flex items-center gap-2">
            <FileUp className="h-4 w-4" />
            <span className="hidden sm:inline">Processing</span>
          </TabsTrigger>
          <TabsTrigger value="visualization" disabled={!currentPath} className="flex items-center gap-2">
            <BarChart4 className="h-4 w-4" />
            <span className="hidden sm:inline">Visualization</span>
          </TabsTrigger>
          <TabsTrigger value="ml-analysis" disabled={!currentPath} className="flex items-center gap-2">
            <BarChart4 className="h-4 w-4" />
            <span className="hidden sm:inline">ML Analysis</span>
          </TabsTrigger>
          <TabsTrigger value="ml-comparison" disabled={!currentPath} className="flex items-center gap-2">
             <BarChart4 className="h-4 w-4" />
            <span className="hidden sm:inline">Comparison</span>
          </TabsTrigger>
        </TabsList>

        <TabsContent value="upload">
          <Card>
            <CardHeader>
              <CardTitle>Upload Data</CardTitle>
              <CardDescription>Upload a CSV file to begin processing your data</CardDescription>
            </CardHeader>
            <CardContent>
              <FileUploader onFileUploaded={handleFileUploaded} />
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="preview">
          {currentPath && (
            <Card>
              <CardHeader>
                <CardTitle>Data Preview</CardTitle>
                <CardDescription>Preview and select columns from your dataset</CardDescription>
              </CardHeader>
              <CardContent>
                <DataPreview
                  path={currentPath}
                  columns={columns}
                  onColumnsSelected={(newPath) => setCurrentPath(newPath)}
                />
              </CardContent>
            </Card>
          )}
        </TabsContent>

        <TabsContent value="statistics">
          {currentPath && (
            <Card>
              <CardHeader>
                <CardTitle>Data Statistics</CardTitle>
                <CardDescription>View statistical information about your dataset</CardDescription>
              </CardHeader>
              <CardContent>
                <DataStatistics path={currentPath} />
              </CardContent>
            </Card>
          )}
        </TabsContent>

        <TabsContent value="processing">
          {currentPath && (
            <Card>
              <CardHeader>
                <CardTitle>Data Processing</CardTitle>
                <CardDescription>Apply transformations to your dataset</CardDescription>
              </CardHeader>
              <CardContent>
                <DataProcessing
                  path={currentPath}
                  columns={columns}
                  onProcessingComplete={(newPath) => {
                    setCurrentPath(newPath)
                  }}
                />
              </CardContent>
            </Card>
          )}
        </TabsContent>

        <TabsContent value="visualization">
          {currentPath && (
            <Card>
              <CardHeader>
                <CardTitle>Data Visualization</CardTitle>
                <CardDescription>Create visualizations from your dataset</CardDescription>
              </CardHeader>
              <CardContent>
                <DataVisualization path={currentPath} columns={columns} />
              </CardContent>
            </Card>
          )}
        </TabsContent>

        <TabsContent value="ml-analysis">
          {currentPath && (
            <Card>
              <CardHeader>
                <CardTitle>ML Analysis</CardTitle>
                <CardDescription>Run Machine Learning algorithms</CardDescription>
              </CardHeader>
              <CardContent>
                <MlAnalysis path={currentPath} columns={columns} />
              </CardContent>
            </Card>
          )}
        </TabsContent>

        <TabsContent value="ml-comparison">
          {currentPath && (
             <Card>
                <CardHeader>
                   <CardTitle>ML Comparison</CardTitle>
                   <CardDescription>Compare performance of different models</CardDescription>
                </CardHeader>
                <CardContent>
                   <MlComparison />
                </CardContent>
             </Card>
          )}
        </TabsContent>
      </Tabs>

      {currentPath && (
        <div className="mt-6 flex justify-end">
          <Button
            variant="outline"
            className="flex items-center gap-2"
            onClick={async () => {
              try {
                const response = await fetch("/api/data/save", {
                  method: "POST",
                  headers: {
                    "Content-Type": "application/json",
                  },
                  body: JSON.stringify({ new_path: currentPath }),
                })

                const data = await response.json()
                if (data.path) {
                  alert(`Data saved successfully to ${data.path}`)
                }
              } catch (error) {
                console.error("Error saving data:", error)
                alert("Error saving data")
              }
            }}
          >
            <Save className="h-4 w-4 mr-2" />
            Save Data
          </Button>
        </div>
      )}
    </main>
  )
}
