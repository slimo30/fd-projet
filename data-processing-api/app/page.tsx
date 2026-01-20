"use client"

import { useState } from "react"
import { Upload, FileUp, Database, BarChart4, Table } from "lucide-react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { useData } from "@/context/DataContext"
import FileUploader from "@/components/file-uploader"
import DataPreview from "@/components/data-preview"
import DataStatistics from "@/components/data-statistics"
import DataProcessing from "@/components/data-processing"
import DataVisualization from "@/components/data-visualization"

export default function Home() {
  const { dataPath, columns, setData, updateProcessedPath } = useData()
  const [activeTab, setActiveTab] = useState("upload")

  const handleFileUploaded = async (path: string, cols: string[]) => {
    setData(path, cols)
    setActiveTab("preview")
  }

  return (
    <main className="container mx-auto py-6 px-4 max-w-7xl">
      <div className="mb-8">
        <h1 className="text-3xl font-bold tracking-tight">Data Processing</h1>
        <p className="text-muted-foreground mt-1">
          Upload, analyze, and transform your data
        </p>
      </div>

      <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
        <TabsList className="grid w-full grid-cols-5 mb-6">
          <TabsTrigger value="upload" className="gap-2">
            <Upload className="h-4 w-4" />
            <span className="hidden sm:inline">Upload</span>
          </TabsTrigger>
          <TabsTrigger value="preview" disabled={!dataPath} className="gap-2">
            <Table className="h-4 w-4" />
            <span className="hidden sm:inline">Preview</span>
          </TabsTrigger>
          <TabsTrigger value="statistics" disabled={!dataPath} className="gap-2">
            <Database className="h-4 w-4" />
            <span className="hidden sm:inline">Statistics</span>
          </TabsTrigger>
          <TabsTrigger value="processing" disabled={!dataPath} className="gap-2">
            <FileUp className="h-4 w-4" />
            <span className="hidden sm:inline">Processing</span>
          </TabsTrigger>
          <TabsTrigger value="visualization" disabled={!dataPath} className="gap-2">
            <BarChart4 className="h-4 w-4" />
            <span className="hidden sm:inline">Visualization</span>
          </TabsTrigger>
        </TabsList>

        <TabsContent value="upload">
          <Card>
            <CardHeader>
              <CardTitle>Upload Data</CardTitle>
              <CardDescription>Upload a CSV file to get started</CardDescription>
            </CardHeader>
            <CardContent>
              <FileUploader onFileUploaded={handleFileUploaded} />
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="preview">
          {dataPath && (
            <Card>
              <CardHeader>
                <CardTitle>Data Preview</CardTitle>
                <CardDescription>Review your data and select columns</CardDescription>
              </CardHeader>
              <CardContent>
                <DataPreview
                  path={dataPath}
                  columns={columns}
                  onColumnsSelected={updateProcessedPath}
                />
              </CardContent>
            </Card>
          )}
        </TabsContent>

        <TabsContent value="statistics">
          {dataPath && (
            <Card>
              <CardHeader>
                <CardTitle>Data Statistics</CardTitle>
                <CardDescription>Statistical analysis of your dataset</CardDescription>
              </CardHeader>
              <CardContent>
                <DataStatistics path={dataPath} />
              </CardContent>
            </Card>
          )}
        </TabsContent>

        <TabsContent value="processing">
          {dataPath && (
            <Card>
              <CardHeader>
                <CardTitle>Data Processing</CardTitle>
                <CardDescription>Clean and transform your data</CardDescription>
              </CardHeader>
              <CardContent>
                <DataProcessing
                  path={dataPath}
                  columns={columns}
                  onProcessingComplete={updateProcessedPath}
                />
              </CardContent>
            </Card>
          )}
        </TabsContent>

        <TabsContent value="visualization">
          {dataPath && (
            <Card>
              <CardHeader>
                <CardTitle>Data Visualization</CardTitle>
                <CardDescription>Create charts from your data</CardDescription>
              </CardHeader>
              <CardContent>
                <DataVisualization path={dataPath} columns={columns} />
              </CardContent>
            </Card>
          )}
        </TabsContent>
      </Tabs>
    </main>
  )
}
