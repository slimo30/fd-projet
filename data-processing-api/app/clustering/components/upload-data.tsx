"use client"

import type React from "react"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Progress } from "@/components/ui/progress"
import { AlertCircle, Upload } from "lucide-react"
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
import { toast } from "sonner"

interface UploadDataProps {
  onUploadSuccess: (path: string) => void
}

export default function UploadData({ onUploadSuccess }: UploadDataProps) {
  const [file, setFile] = useState<File | null>(null)
  const [uploading, setUploading] = useState(false)
  const [progress, setProgress] = useState(0)
  const [error, setError] = useState<string | null>(null)
  const [dataPreview, setDataPreview] = useState<any[] | null>(null)

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0])
      setError(null)
    }
  }

  const handleUpload = async () => {
    if (!file) {
      setError("Please select a file to upload")
      return
    }

    setUploading(true)
    setProgress(0)
    setError(null)

    // Create a FormData object to send the file
    const formData = new FormData()
    formData.append("file", file)

    try {
      // Simulate progress for better UX
      const progressInterval = setInterval(() => {
        setProgress((prev) => {
          const newProgress = prev + 5
          return newProgress > 90 ? 90 : newProgress
        })
      }, 100)

      const response = await fetch("http://localhost:5000/upload-data-clustering", {
        method: "POST",
        body: formData,
      })

      clearInterval(progressInterval)
      setProgress(100)

      if (!response.ok) {
        const errorData = await response.json()
        throw new Error(errorData.error || "Failed to upload file")
      }

      const data = await response.json()
      setDataPreview(data.head)
      onUploadSuccess(data.path)
      toast.success("File uploaded successfully")
    } catch (err: any) {
      setError(err.message || "An error occurred during upload")
      toast.error("Upload failed", {
        description: err.message,
      })
    } finally {
      setUploading(false)
    }
  }

  return (
    <div className="space-y-4">
      <div className="flex items-center gap-4">
        <Input type="file" accept=".csv" onChange={handleFileChange} disabled={uploading} className="flex-1" />
        <Button onClick={handleUpload} disabled={!file || uploading} className="flex gap-2 items-center">
          <Upload size={16} />
          Upload
        </Button>
      </div>

      {uploading && (
        <div className="space-y-2">
          <div className="text-sm text-muted-foreground">Uploading {file?.name}...</div>
          <Progress value={progress} className="h-2" />
        </div>
      )}

      {error && (
        <Alert variant="destructive">
          <AlertCircle className="h-4 w-4" />
          <AlertTitle>Error</AlertTitle>
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}

      {dataPreview && (
        <div className="mt-4">
          <h3 className="text-lg font-medium mb-2">Data Preview</h3>
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-gray-200">
              <thead className="bg-gray-50">
                <tr>
                  {Object.keys(dataPreview[0]).map((key) => (
                    <th
                      key={key}
                      className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider"
                    >
                      {key}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-gray-200">
                {dataPreview.map((row, i) => (
                  <tr key={i}>
                    {Object.values(row).map((value: any, j) => (
                      <td key={j} className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                        {value}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  )
}
