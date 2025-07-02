"use client"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group"
import { Label } from "@/components/ui/label"
import { AlertCircle, Check } from "lucide-react"
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
import { toast } from "sonner"

interface AlgorithmSelectionProps {
  filePath: string
  onAlgorithmSelect: (algorithm: string) => void
}

export default function AlgorithmSelection({ filePath, onAlgorithmSelect }: AlgorithmSelectionProps) {
  const [algorithm, setAlgorithm] = useState("kmeans")
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [success, setSuccess] = useState(false)

  const handleSubmit = async () => {
    setLoading(true)
    setError(null)
    setSuccess(false)

    try {
      const response = await fetch("http://localhost:5000/clustering/select", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          path: filePath,
          algorithm,
        }),
      })

      if (!response.ok) {
        const errorData = await response.json()
        throw new Error(errorData.error || "Failed to select algorithm")
      }

      setSuccess(true)
      onAlgorithmSelect(algorithm)
      toast.success(`${algorithm.toUpperCase()} algorithm selected`)
    } catch (err: any) {
      setError(err.message || "An error occurred")
      toast.error("Selection failed", {
        description: err.message,
      })
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="space-y-6">
      <RadioGroup value={algorithm} onValueChange={setAlgorithm} className="grid grid-cols-2 gap-4">
        <div className="flex items-center space-x-2 border p-4 rounded-md">
          <RadioGroupItem value="kmeans" id="kmeans" />
          <Label htmlFor="kmeans" className="font-medium">
            K-Means
          </Label>
        </div>
        <div className="flex items-center space-x-2 border p-4 rounded-md">
          <RadioGroupItem value="kmedoids" id="kmedoids" />
          <Label htmlFor="kmedoids" className="font-medium">
            K-Medoids
          </Label>
        </div>
        <div className="flex items-center space-x-2 border p-4 rounded-md">
          <RadioGroupItem value="agnes" id="agnes" />
          <Label htmlFor="agnes" className="font-medium">
            Hierarchical (AGNES)
          </Label>
        </div>
        <div className="flex items-center space-x-2 border p-4 rounded-md">
          <RadioGroupItem value="dbscan" id="dbscan" />
          <Label htmlFor="dbscan" className="font-medium">
            DBSCAN
          </Label>
        </div>
      </RadioGroup>

      <Button onClick={handleSubmit} disabled={loading} className="w-full">
        {loading ? "Selecting..." : "Select Algorithm"}
      </Button>

      {error && (
        <Alert variant="destructive">
          <AlertCircle className="h-4 w-4" />
          <AlertTitle>Error</AlertTitle>
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}

      {success && (
        <Alert className="bg-green-50 border-green-200">
          <Check className="h-4 w-4 text-green-600" />
          <AlertTitle className="text-green-800">Success</AlertTitle>
          <AlertDescription className="text-green-700">
            Algorithm {algorithm.toUpperCase()} selected successfully
          </AlertDescription>
        </Alert>
      )}
    </div>
  )
}
