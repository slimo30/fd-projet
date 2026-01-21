"use client"

import { useState } from "react"
import { Play, Loader2, Gauge, AlertCircle, CheckCircle2 } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { cnnApi, ApiError } from "@/lib/api"
import { toast } from "sonner"
import { Progress } from "@/components/ui/progress"

interface CNNTrainingProps {
    dataset: string
    onTrainingComplete: () => void
}

export default function CNNTraining({ dataset, onTrainingComplete }: CNNTrainingProps) {
    const [loading, setLoading] = useState(false)
    const [params, setParams] = useState({
        epochs: 10,
        batch_size: 32,
        test_size: 0.2
    })
    const [results, setResults] = useState<any>(null)

    const handleTrain = async () => {
        try {
            setLoading(true)
            setResults(null)
            toast.info("Training started. This may take a while...")

            const result = await cnnApi.train(
                dataset,
                params.epochs,
                params.batch_size,
                params.test_size
            )

            setResults(result)
            onTrainingComplete()
            toast.success("Model trained successfully!")
        } catch (error) {
            console.error("Error training model:", error)
            toast.error(error instanceof Error ? error.message : "Training failed")
        } finally {
            setLoading(false)
        }
    }

    return (
        <div className="space-y-8">
            <div className="grid md:grid-cols-3 gap-6">
                <div className="grid gap-2">
                    <Label htmlFor="epochs">Epochs</Label>
                    <Input
                        id="epochs"
                        type="number"
                        min="1"
                        max="100"
                        value={params.epochs}
                        onChange={(e) => setParams({ ...params, epochs: parseInt(e.target.value) || 10 })}
                    />
                    <p className="text-xs text-muted-foreground">Number of passes through dataset</p>
                </div>

                <div className="grid gap-2">
                    <Label htmlFor="batch">Batch Size</Label>
                    <Input
                        id="batch"
                        type="number"
                        min="1"
                        value={params.batch_size}
                        onChange={(e) => setParams({ ...params, batch_size: parseInt(e.target.value) || 32 })}
                    />
                    <p className="text-xs text-muted-foreground">Samples per gradient update</p>
                </div>

                <div className="flex items-end">
                    <Button
                        onClick={handleTrain}
                        disabled={loading}
                        className="w-full"
                        size="lg"
                    >
                        {loading ? (
                            <>
                                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                                Training...
                            </>
                        ) : (
                            <>
                                <Play className="mr-2 h-4 w-4" />
                                Start Training
                            </>
                        )}
                    </Button>
                </div>
            </div>

            {loading && (
                <div className="space-y-4 py-8 border rounded-lg p-6 bg-muted/20">
                    <div className="flex items-center gap-4">
                        <Loader2 className="h-8 w-8 animate-spin text-primary" />
                        <div className="space-y-1">
                            <h4 className="font-semibold">Training in progress...</h4>
                            <p className="text-sm text-muted-foreground">
                                Depending on the dataset and epochs, this can take from a few seconds to minutes.
                            </p>
                        </div>
                    </div>
                    <Progress value={undefined} className="w-full" />
                </div>
            )}

            {results && (
                <div className="space-y-6 animate-in fade-in slide-in-from-bottom-4">
                    <div className="grid md:grid-cols-4 gap-4">
                        <div className="border rounded-lg p-4 bg-green-50 border-green-200">
                            <div className="text-sm font-medium text-green-800 mb-1">Accuracy</div>
                            <div className="text-2xl font-bold text-green-900">
                                {(results.metrics.accuracy * 100).toFixed(2)}%
                            </div>
                        </div>
                        <div className="border rounded-lg p-4 bg-blue-50 border-blue-200">
                            <div className="text-sm font-medium text-blue-800 mb-1">Train Loss</div>
                            <div className="text-2xl font-bold text-blue-900">
                                {results.metrics.train_loss.toFixed(4)}
                            </div>
                        </div>
                        <div className="border rounded-lg p-4 bg-purple-50 border-purple-200">
                            <div className="text-sm font-medium text-purple-800 mb-1">Epochs</div>
                            <div className="text-2xl font-bold text-purple-900">
                                {results.metrics.epochs_trained}
                            </div>
                        </div>
                        <div className="border rounded-lg p-4 bg-orange-50 border-orange-200">
                            <div className="text-sm font-medium text-orange-800 mb-1">Parameters</div>
                            <div className="text-2xl font-bold text-orange-900">
                                {results.metrics.total_params.toLocaleString()}
                            </div>
                        </div>
                    </div>

                    <div className="grid md:grid-cols-2 gap-6">
                        <div className="border rounded-lg p-4 bg-white">
                            <h4 className="font-semibold mb-4 text-center">Training History</h4>
                            <div className="relative aspect-video">
                                <img
                                    src={`http://127.0.0.1:5001${results.history_plot_url}`}
                                    alt="Training History"
                                    className="w-full h-full object-contain"
                                />
                            </div>
                        </div>
                        <div className="border rounded-lg p-4 bg-white">
                            <h4 className="font-semibold mb-4 text-center">Confusion Matrix</h4>
                            <div className="relative aspect-video">
                                <img
                                    src={`http://127.0.0.1:5001${results.confusion_matrix_plot_url}`}
                                    alt="Confusion Matrix"
                                    className="w-full h-full object-contain"
                                />
                            </div>
                        </div>
                    </div>

                    <div className="bg-slate-900 text-slate-50 p-4 rounded-lg overflow-x-auto">
                        <pre className="text-xs font-mono">
                            {results.model_summary}
                        </pre>
                    </div>
                </div>
            )}
        </div>
    )
}
