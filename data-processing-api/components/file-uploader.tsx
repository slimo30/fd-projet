"use client"

import type React from "react"

import { useState } from "react"
import { Upload, Loader2, Check, FileText } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Progress } from "@/components/ui/progress"
import { dataApi, ApiError } from "@/lib/api"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"

interface FileUploaderProps {
  onFileUploaded: (path: string, columns: string[]) => void
}

export default function FileUploader({ onFileUploaded }: FileUploaderProps) {
  const [file, setFile] = useState<File | null>(null)
  const [isUploading, setIsUploading] = useState(false)
  const [uploadProgress, setUploadProgress] = useState(0)
  const [error, setError] = useState<string | null>(null)

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      setFile(e.target.files[0])
      setError(null)
    }
  }

  const handleUpload = async () => {
    if (!file) {
      setError("Please select a file first")
      return
    }

    setIsUploading(true)
    setUploadProgress(0)
    setError(null)

    try {
      const progressInterval = setInterval(() => {
        setUploadProgress((prev) => {
          const newProgress = prev + 5
          return newProgress > 90 ? 90 : newProgress
        })
      }, 100)

      const data = await dataApi.uploadData(file)

      clearInterval(progressInterval)
      setUploadProgress(100)

      if (data.path && data.columns) {
        onFileUploaded(data.path, data.columns)
      } else {
        throw new Error("No path or columns returned from server")
      }
    } catch (error) {
      console.error("Error uploading file:", error)
      if (error instanceof ApiError) {
        setError(`Upload failed: ${error.message}`)
      } else {
        setError(error instanceof Error ? error.message : "An unknown error occurred")
      }
    } finally {
      setIsUploading(false)
    }
  }

  const handleCreateSample = async (clean: boolean) => {
    setIsUploading(true)
    setUploadProgress(0)
    setError(null)

    try {
      const progressInterval = setInterval(() => {
        setUploadProgress((prev) => {
          const newProgress = prev + 10
          return newProgress > 90 ? 90 : newProgress
        })
      }, 100)

      const data = await dataApi.createSample(clean)

      clearInterval(progressInterval)
      setUploadProgress(100)

      if (data.path && data.columns) {
        onFileUploaded(data.path, data.columns)
      } else {
        throw new Error("No path or columns returned from server")
      }
    } catch (error) {
      console.error("Error creating sample data:", error)
      if (error instanceof ApiError) {
        setError(`Failed to create sample: ${error.message}`)
      } else {
        setError(error instanceof Error ? error.message : "An unknown error occurred")
      }
    } finally {
      setIsUploading(false)
    }
  }

  return (
    <div className="space-y-6">
      {/* File Upload Section */}
      <div className="border-2 border-dashed rounded-lg p-8 text-center hover:border-gray-400 transition-colors">
        <div className="flex flex-col items-center justify-center space-y-3">
          <Upload className="h-12 w-12 text-gray-400" />
          <div>
            <h3 className="text-lg font-medium">Upload CSV File</h3>
            <p className="text-sm text-muted-foreground mt-1">Select a file from your computer</p>
          </div>

          <input
            type="file"
            id="file-upload"
            className="hidden"
            accept=".csv"
            onChange={handleFileChange}
            disabled={isUploading}
          />

          <Button
            variant="outline"
            onClick={() => document.getElementById("file-upload")?.click()}
            disabled={isUploading}
            size="lg"
          >
            Select File
          </Button>

          {file && (
            <div className="mt-2 p-3 bg-muted rounded-md border">
              <p className="text-sm font-medium">
                {file.name} ({(file.size / 1024).toFixed(2)} KB)
              </p>
            </div>
          )}
        </div>
      </div>

      {error && (
        <div className="bg-destructive/10 text-destructive p-3 rounded-md text-sm border border-destructive/20">
          {error}
        </div>
      )}

      {isUploading && (
        <div className="space-y-2">
          <Progress value={uploadProgress} className="h-2" />
          <p className="text-sm text-center text-muted-foreground">
            {uploadProgress < 100 ? "Uploading..." : "Processing..."}
          </p>
        </div>
      )}

      <Button 
        className="w-full" 
        onClick={handleUpload} 
        disabled={!file || isUploading} 
        size="lg"
      >
        {isUploading ? (
          <>
            <Loader2 className="mr-2 h-4 w-4 animate-spin" />
            Uploading...
          </>
        ) : (
          "Upload File"
        )}
      </Button>

{/* Divider */}
<div className="relative">
  <div className="absolute inset-0 flex items-center">
    <span className="w-full border-t" />
  </div>
  <div className="relative flex justify-center text-xs uppercase">
    <span className="bg-background px-2 text-muted-foreground">Or use sample data</span>
  </div>
</div>

{/* Sample Data Options */}
<div className="grid grid-cols-1 md:grid-cols-2 gap-4">
  <Card className="hover:shadow-md transition-shadow">
    <CardHeader className="pb-3">
      <div className="flex items-center justify-between">
        <CardTitle className="text-base flex items-center gap-2">
          <Check className="h-4 w-4" />
          Clean Dataset
        </CardTitle>
        <Badge variant="secondary">Normalized</Badge>
      </div>
    </CardHeader>
    <CardContent className="space-y-3">
      <CardDescription className="text-sm">
        Numeric, normalized data ready for machine learning
      </CardDescription>
      <ul className="text-xs space-y-1.5 text-muted-foreground">
        <li>• All numeric features (0-1 range)</li>
        <li>• Pre-normalized values</li>
        <li>• No missing data</li>
        <li>• Binary target variable</li>
      </ul>
      <Button
        variant="outline"
        className="w-full"
        onClick={() => handleCreateSample(true)}
        disabled={isUploading}
      >
        Create Clean Dataset
      </Button>
    </CardContent>
  </Card>

  <Card className="hover:shadow-md transition-shadow">
    <CardHeader className="pb-3">
      <div className="flex items-center justify-between">
        <CardTitle className="text-base flex items-center gap-2">
          <FileText className="h-4 w-4" />
          Raw Dataset
        </CardTitle>
        <Badge variant="outline">Preprocessing Required</Badge>
      </div>
    </CardHeader>
    <CardContent className="space-y-3">
      <CardDescription className="text-sm">
        Mixed data types requiring preprocessing
      </CardDescription>
      <ul className="text-xs space-y-1.5 text-muted-foreground">
        <li>• Mixed numeric & categorical</li>
        <li>• Contains missing values</li>
        <li>• Requires normalization</li>
        <li>• Practice preprocessing</li>
      </ul>
      <Button
        variant="outline"
        className="w-full"
        onClick={() => handleCreateSample(false)}
        disabled={isUploading}
      >
        Create Raw Dataset
      </Button>
    </CardContent>
  </Card>
</div>

      <div className="text-xs text-center text-muted-foreground bg-muted/50 rounded-md p-3">
        <strong>Note:</strong> Clean dataset is ready for immediate analysis. Raw dataset requires preprocessing steps.
      </div>
    </div>
  )
}
