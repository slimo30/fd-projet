"use client"

import { useState, useEffect } from "react"
import { Check, Database, Image as ImageIcon, Loader2 } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { cnnApi, ApiError } from "@/lib/api"
import { toast } from "sonner"
import Image from "next/image"

interface DatasetSelectionProps {
    selectedDataset: string
    onSelect: (dataset: string) => void
}

export default function DatasetSelection({ selectedDataset, onSelect }: DatasetSelectionProps) {
    const [datasets, setDatasets] = useState<Record<string, any>>({})
    const [loading, setLoading] = useState(true)
    const [infoLoading, setInfoLoading] = useState(false)
    const [datasetInfo, setDatasetInfo] = useState<any>(null)
    const [samplePlotUrl, setSamplePlotUrl] = useState<string | null>(null)
    const [sampleLoading, setSampleLoading] = useState(false)

    useEffect(() => {
        fetchDatasets()
    }, [])

    useEffect(() => {
        if (selectedDataset) {
            fetchDatasetInfo(selectedDataset)
        }
    }, [selectedDataset])

    const fetchDatasets = async () => {
        try {
            setLoading(true)
            const result = await cnnApi.getDatasets()
            setDatasets(result.datasets)
        } catch (error) {
            console.error("Error fetching datasets:", error)
            toast.error("Failed to load datasets")
        } finally {
            setLoading(false)
        }
    }

    const fetchDatasetInfo = async (name: string) => {
        try {
            setInfoLoading(true)
            const result = await cnnApi.getDatasetInfo(name)
            setDatasetInfo(result.info)
            fetchSamples(name)
        } catch (error) {
            console.error("Error fetching dataset info:", error)
            toast.error("Failed to load dataset info")
        } finally {
            setInfoLoading(false)
        }
    }

    const fetchSamples = async (name: string) => {
        try {
            setSampleLoading(true)
            const result = await cnnApi.visualizeSamples(name)
            setSamplePlotUrl(result.plot_url)
        } catch (error) {
            console.error("Error fetching samples:", error)
            toast.error("Failed to load samples")
        } finally {
            setSampleLoading(false)
        }
    }

    if (loading) {
        return (
            <div className="flex justify-center py-12">
                <Loader2 className="h-8 w-8 animate-spin text-primary" />
            </div>
        )
    }

    return (
        <div className="space-y-6">
            <div className="flex flex-col sm:flex-row gap-4 items-start sm:items-center">
                <div className="w-full sm:w-[300px]">
                    <label className="text-sm font-medium mb-1.5 block">Select Dataset</label>
                    <Select value={selectedDataset} onValueChange={onSelect}>
                        <SelectTrigger>
                            <SelectValue placeholder="Choose a dataset" />
                        </SelectTrigger>
                        <SelectContent>
                            {Object.entries(datasets).map(([key, info]: [string, any]) => (
                                <SelectItem key={key} value={key}>
                                    {info.name}
                                </SelectItem>
                            ))}
                        </SelectContent>
                    </Select>
                </div>

                {datasetInfo && (
                    <div className="flex items-center gap-4 mt-2 sm:mt-0 text-sm text-gray-500">
                        <div className="flex items-center gap-1.5">
                            <Database className="h-4 w-4" />
                            <span>{datasetInfo.total_samples.toLocaleString()} samples</span>
                        </div>
                        <div className="flex items-center gap-1.5">
                            <ImageIcon className="h-4 w-4" />
                            <span>{datasetInfo.shape[0]}×{datasetInfo.shape[1]} px</span>
                        </div>
                    </div>
                )}
            </div>

            {infoLoading ? (
                <div className="flex justify-center py-12">
                    <Loader2 className="h-8 w-8 animate-spin text-primary" />
                </div>
            ) : datasetInfo ? (
                <div className="space-y-6">
                    <div className="prose max-w-none">
                        <h3 className="text-lg font-semibold">About this Dataset</h3>
                        <p className="text-muted-foreground">{datasetInfo.description}</p>
                    </div>

                    <div className="grid md:grid-cols-2 gap-6">
                        <div className="bg-muted/50 p-4 rounded-lg">
                            <h4 className="font-semibold mb-2">Class Distribution</h4>
                            <div className="space-y-2">
                                {Object.entries(datasetInfo.train_distribution).slice(0, 5).map(([cls, count]: [string, any]) => (
                                    <div key={cls} className="flex justify-between items-center text-sm">
                                        <span>{cls}</span>
                                        <span className="font-mono text-muted-foreground">{count}</span>
                                    </div>
                                ))}
                                {Object.keys(datasetInfo.train_distribution).length > 5 && (
                                    <div className="text-xs text-muted-foreground pt-1">
                                        ...and {Object.keys(datasetInfo.train_distribution).length - 5} more classes
                                    </div>
                                )}
                            </div>
                        </div>

                        <div className="bg-muted/50 p-4 rounded-lg">
                            <h4 className="font-semibold mb-2">Dataset Specs</h4>
                            <dl className="space-y-1 text-sm">
                                <div className="flex justify-between">
                                    <dt className="text-muted-foreground">Image Shape</dt>
                                    <dd className="font-mono">({datasetInfo.image_shape.join(', ')})</dd>
                                </div>
                                <div className="flex justify-between">
                                    <dt className="text-muted-foreground">Check Channels</dt>
                                    <dd>{datasetInfo.image_shape[2] === 1 ? 'Grayscale' : 'RGB Color'}</dd>
                                </div>
                                <div className="flex justify-between">
                                    <dt className="text-muted-foreground">Classes</dt>
                                    <dd>{datasetInfo.num_classes}</dd>
                                </div>
                                <div className="flex justify-between">
                                    <dt className="text-muted-foreground">Train/Test Split</dt>
                                    <dd>{datasetInfo.train_samples} / {datasetInfo.test_samples}</dd>
                                </div>
                            </dl>
                        </div>
                    </div>

                    <div className="border rounded-lg p-4 bg-white">
                        <h4 className="font-semibold mb-4">Sample Images</h4>
                        {sampleLoading ? (
                            <div className="flex justify-center py-8">
                                <Loader2 className="h-8 w-8 animate-spin text-gray-400" />
                            </div>
                        ) : samplePlotUrl ? (
                            <div className="relative w-full aspect-[21/9] min-h-[300px]">
                                <img
                                    src={`http://127.0.0.1:5001${samplePlotUrl}`}
                                    alt="Dataset Samples"
                                    className="rounded-md object-contain w-full h-full"
                                />
                            </div>
                        ) : (
                            <div className="text-center py-8 text-muted-foreground">
                                Failed to load samples
                            </div>
                        )}
                        <div className="flex justify-end mt-2">
                            <Button variant="outline" size="sm" onClick={() => fetchSamples(selectedDataset)}>
                                <ImageIcon className="mr-2 h-4 w-4" />
                                Refresh Samples
                            </Button>
                        </div>
                    </div>
                </div>
            ) : null}
        </div>
    )
}
