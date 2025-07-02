"use client"

import type React from "react"

import { useState } from "react"
import { Upload, Loader2 } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Progress } from "@/components/ui/progress"

interface FileUploaderProps {
  onFileUploaded: (path: string) => void
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

    // Create a FormData object to send the file
    const formData = new FormData()
    formData.append("file", file)

    try {
      // Simulate progress for better UX
      const progressInterval = setInterval(() => {
        setUploadProgress((prev) => {
          const newProgress = prev + 5
          return newProgress > 90 ? 90 : newProgress
        })
      }, 100)

      const response = await fetch("/api/upload-data", {
        method: "POST",
        body: formData,
      })

      clearInterval(progressInterval)
      setUploadProgress(100)

      if (!response.ok) {
        throw new Error(`Upload failed: ${response.statusText}`)
      }

      const data = await response.json()

      if (data.path) {
        onFileUploaded(data.path)
      } else {
        throw new Error("No path returned from server")
      }
    } catch (error) {
      console.error("Error uploading file:", error)
      setError(error instanceof Error ? error.message : "An unknown error occurred")
    } finally {
      setIsUploading(false)
    }
  }

  const handleCreateSample = async () => {
    setIsUploading(true)
    setUploadProgress(0)

    try {
      // Simulate progress for better UX
      const progressInterval = setInterval(() => {
        setUploadProgress((prev) => {
          const newProgress = prev + 10
          return newProgress > 90 ? 90 : newProgress
        })
      }, 100)

      const response = await fetch("/api/create-sample", {
        method: "POST",
      })

      clearInterval(progressInterval)
      setUploadProgress(100)

      if (!response.ok) {
        throw new Error(`Failed to create sample: ${response.statusText}`)
      }

      const data = await response.json()

      if (data.path) {
        onFileUploaded(data.path)
      } else {
        throw new Error("No path returned from server")
      }
    } catch (error) {
      console.error("Error creating sample data:", error)
      setError(error instanceof Error ? error.message : "An unknown error occurred")
    } finally {
      setIsUploading(false)
    }
  }

  return (
    <div className="space-y-6">
      <div className="border-2 border-dashed border-gray-300 rounded-lg p-6 text-center">
        <div className="flex flex-col items-center justify-center space-y-2">
          <Upload className="h-10 w-10 text-gray-400" />
          <h3 className="text-lg font-medium">Drag and drop your CSV file here</h3>
          <p className="text-sm text-gray-500">or click to browse files</p>

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
          >
            Select File
          </Button>

          {file && (
            <p className="text-sm font-medium">
              Selected: {file.name} ({(file.size / 1024).toFixed(2)} KB)
            </p>
          )}
        </div>
      </div>

      {error && <div className="bg-red-50 text-red-500 p-3 rounded-md text-sm">{error}</div>}

      {isUploading && (
        <div className="space-y-2">
          <Progress value={uploadProgress} className="h-2" />
          <p className="text-sm text-center text-gray-500">{uploadProgress < 100 ? "Uploading..." : "Processing..."}</p>
        </div>
      )}

      <div className="flex flex-col sm:flex-row gap-4">
        <Button className="flex-1" onClick={handleUpload} disabled={!file || isUploading}>
          {isUploading ? (
            <>
              <Loader2 className="mr-2 h-4 w-4 animate-spin" />
              Uploading...
            </>
          ) : (
            "Upload File"
          )}
        </Button>

        <Button variant="outline" className="flex-1" onClick={handleCreateSample} disabled={isUploading}>
          Create Sample Data
        </Button>
      </div>
    </div>
  )
}
