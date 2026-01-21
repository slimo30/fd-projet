"use client"

import { useState, useEffect } from "react"
import { Loader2, RefreshCw, AlertTriangle } from "lucide-react"
import { Button } from "@/components/ui/button"
import { cnnApi } from "@/lib/api"
import { toast } from "sonner"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"

interface PredictionsViewProps {
    dataset: string
}

export default function PredictionsView({ dataset }: PredictionsViewProps) {
    const [loading, setLoading] = useState(false)
    const [predictions, setPredictions] = useState<any>(null)
    const [errors, setErrors] = useState<any>(null)
    const [mode, setMode] = useState("samples")

    useEffect(() => {
        fetchPredictions()
        fetchErrors()
    }, [dataset])

    const fetchPredictions = async () => {
        try {
            setLoading(true)
            const result = await cnnApi.getPredictions()
            setPredictions(result)
        } catch (error) {
            console.error("Error fetching predictions:", error)
        } finally {
            setLoading(false)
        }
    }

    const fetchErrors = async () => {
        try {
            const result = await cnnApi.getErrorAnalysis()
            setErrors(result)
        } catch (error) {
            console.error("Error fetching error analysis:", error)
        }
    }

    return (
        <div className="space-y-6">
            <div className="flex justify-between items-center">
                <Tabs value={mode} onValueChange={setMode} className="w-[400px]">
                    <TabsList>
                        <TabsTrigger value="samples" className="flex-1">Test Samples</TabsTrigger>
                        <TabsTrigger value="errors" className="flex-1">Error Analysis</TabsTrigger>
                    </TabsList>
                </Tabs>

                <Button
                    variant="outline"
                    size="sm"
                    onClick={() => {
                        if (mode === 'samples') fetchPredictions()
                        else fetchErrors()
                    }}
                    disabled={loading}
                >
                    <RefreshCw className={`mr-2 h-4 w-4 ${loading ? 'animate-spin' : ''}`} />
                    Refresh
                </Button>
            </div>

            {mode === 'samples' && (
                <div className="space-y-6 animate-in fade-in">
                    {predictions ? (
                        <div className="border rounded-lg p-4 bg-white">
                            <h4 className="font-semibold mb-4 text-center">Model Predictions on Test Set</h4>
                            <div className="relative w-full aspect-[21/9] min-h-[300px]">
                                <img
                                    src={`http://127.0.0.1:5001${predictions.plot_url}`}
                                    alt="Predictions"
                                    className="rounded-md object-contain w-full h-full"
                                />
                            </div>
                            <div className="grid grid-cols-2 md:grid-cols-5 gap-4 mt-6">
                                {predictions.predictions.map((pred: any, i: number) => (
                                    <div key={i} className={`text-xs p-2 rounded border ${pred.correct ? 'bg-green-50 border-green-200' : 'bg-red-50 border-red-200'}`}>
                                        <div className="font-semibold mb-1">Sample #{pred.index}</div>
                                        <div>True: {pred.true_label}</div>
                                        <div>Pred: {pred.predicted_label}</div>
                                        <div className="mt-1 font-mono">{(pred.confidence * 100).toFixed(1)}% conf</div>
                                    </div>
                                ))}
                            </div>
                        </div>
                    ) : (
                        <div className="flex justify-center p-8">
                            {loading ? <Loader2 className="h-8 w-8 animate-spin" /> : <span>No predictions available</span>}
                        </div>
                    )}
                </div>
            )}

            {mode === 'errors' && (
                <div className="space-y-6 animate-in fade-in">
                    {errors ? (
                        errors.num_errors > 0 ? (
                            <div className="space-y-4">
                                <div className="border rounded-lg p-4 bg-red-50 border-red-200 flex items-start gap-4">
                                    <AlertTriangle className="h-5 w-5 text-red-600 mt-0.5" />
                                    <div>
                                        <h4 className="font-semibold text-red-900">Found {errors.num_errors} Misclassified Samples</h4>
                                        <p className="text-sm text-red-800">
                                            Error Rate: {(errors.error_rate * 100).toFixed(2)}% on test set
                                        </p>
                                    </div>
                                </div>

                                <div className="border rounded-lg p-4 bg-white">
                                    <h4 className="font-semibold mb-4 text-center">Misclassified Examples</h4>
                                    <div className="relative w-full aspect-[21/9] min-h-[300px]">
                                        <img
                                            src={`http://127.0.0.1:5001${errors.plot_url}`}
                                            alt="Errors"
                                            className="rounded-md object-contain w-full h-full"
                                        />
                                    </div>
                                </div>
                            </div>
                        ) : (
                            <div className="text-center py-12 bg-green-50 rounded-lg border border-green-200">
                                <div className="text-green-600 font-semibold text-lg mb-2">Perfect Score! 🎉</div>
                                <p className="text-green-800">The model classified all test samples correctly.</p>
                            </div>
                        )
                    ) : (
                        <div className="flex justify-center p-8">
                            {loading ? <Loader2 className="h-8 w-8 animate-spin" /> : <span>No error analysis available</span>}
                        </div>
                    )}
                </div>
            )}
        </div>
    )
}
