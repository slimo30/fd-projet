"use client";

import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Progress } from "@/components/ui/progress";
import { AlertCircle, Upload } from "lucide-react";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { toast } from "sonner";
import { dataApi, ApiError } from "@/lib/api";

interface UploadDataProps {
  onUploadSuccess: (path: string, columns: string[]) => void;
}

export default function UploadData({ onUploadSuccess }: UploadDataProps) {
  const [file, setFile] = useState<File | null>(null);
  const [uploading, setUploading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [dataPreview, setDataPreview] = useState<any[] | null>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0]);
      setError(null);
    }
  };

  const handleUpload = async () => {
    if (!file) {
      setError("Please select a file to upload");
      return;
    }

    setUploading(true);
    setProgress(0);
    setError(null);

    try {
      const progressInterval = setInterval(() => {
        setProgress((prev) => {
          const newProgress = prev + 5;
          return newProgress > 90 ? 90 : newProgress;
        });
      }, 100);

      const data = await dataApi.uploadData(file);

      clearInterval(progressInterval);
      setProgress(100);

      setDataPreview(data.head);
      onUploadSuccess(data.path, data.columns);
      toast.success("File uploaded successfully");
    } catch (err: any) {
      const errorMessage = err instanceof ApiError ? err.message : (err.message || "An error occurred during upload");
      setError(errorMessage);
      toast.error("Upload failed", {
        description: errorMessage,
      });
    } finally {
      setUploading(false);
    }
  };

  const handleCreateSample = async () => {
    setUploading(true);
    setProgress(0);
    setError(null);

    try {
      const progressInterval = setInterval(() => {
        setProgress((prev) => {
          const newProgress = prev + 10;
          return newProgress > 90 ? 90 : newProgress;
        });
      }, 100);

      const data = await dataApi.createSample();

      clearInterval(progressInterval);
      setProgress(100);

      setDataPreview(data.head);
      onUploadSuccess(data.path, data.columns);
      toast.success("Sample data created successfully");
    } catch (err: any) {
      const errorMessage = err instanceof ApiError ? err.message : (err.message || "An error occurred");
      setError(errorMessage);
      toast.error("Failed to create sample", {
        description: errorMessage,
      });
    } finally {
      setUploading(false);
    }
  };

  return (
    <div className="space-y-4">
      <div className="flex items-center gap-4">
        <Input type="file" accept=".csv" onChange={handleFileChange} disabled={uploading} className="flex-1" />
        <Button onClick={handleUpload} disabled={!file || uploading} className="flex gap-2 items-center">
          <Upload size={16} />
          Upload
        </Button>
      </div>

      <div className="flex justify-center">
        <Button variant="outline" onClick={handleCreateSample} disabled={uploading}>
          Or Create Sample Data
        </Button>
      </div>

      {uploading && (
        <div className="space-y-2">
          <div className="text-sm text-muted-foreground">
            {file ? `Uploading ${file.name}...` : "Creating sample data..."}
          </div>
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
          <div className="overflow-x-auto border rounded-md">
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
                        {value !== null && value !== undefined ? String(value) : "N/A"}
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
  );
}
