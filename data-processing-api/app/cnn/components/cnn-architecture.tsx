"use client"

import { useState, useEffect } from "react"
import { Loader2, Layers, Cpu } from "lucide-react"
import { cnnApi } from "@/lib/api"
import { ScrollArea } from "@/components/ui/scroll-area"

interface CNNArchitectureProps {
    dataset: string
}

export default function CNNArchitecture({ dataset }: CNNArchitectureProps) {
    const [loading, setLoading] = useState(false)
    const [arch, setArch] = useState<any>(null)

    useEffect(() => {
        fetchArchitecture()
    }, [dataset])

    const fetchArchitecture = async () => {
        try {
            setLoading(true)
            const result = await cnnApi.getArchitecture(dataset)
            setArch(result)
        } catch (error) {
            console.error("Error fetching architecture:", error)
        } finally {
            setLoading(false)
        }
    }

    if (loading) return <div className="flex justify-center p-12"><Loader2 className="animate-spin text-primary" /></div>
    if (!arch) return null

    return (
        <div className="space-y-6">
            <div className="grid md:grid-cols-2 gap-6">
                <div className="space-y-4">
                    <h3 className="text-lg font-semibold flex items-center gap-2">
                        <Layers className="h-5 w-5" />
                        Layer Details
                    </h3>
                    <ScrollArea className="h-[400px] border rounded-md p-4">
                        <div className="space-y-4">
                            {arch.layers.map((layer: any, i: number) => (
                                <div key={i} className="border rounded-lg p-3 bg-card hover:bg-accent/50 transition-colors">
                                    <div className="flex justify-between items-start mb-2">
                                        <span className="font-semibold text-primary">{layer.type}</span>
                                        <span className="text-xs font-mono bg-muted px-1.5 py-0.5 rounded">{layer.name}</span>
                                    </div>
                                    <div className="grid grid-cols-2 gap-x-4 gap-y-1 text-sm text-muted-foreground">
                                        <div>Output: <span className="font-mono text-foreground">{layer.output_shape}</span></div>
                                        <div>Params: <span className="font-mono text-foreground">{layer.params.toLocaleString()}</span></div>
                                        {layer.activation && <div>Activation: <span className="text-foreground capitalize">{layer.activation}</span></div>}
                                        {layer.filters && <div>Filters: <span className="text-foreground">{layer.filters}</span></div>}
                                        {layer.kernel_size && <div>Kernel: <span className="text-foreground">{JSON.stringify(layer.kernel_size)}</span></div>}
                                        {layer.pool_size && <div>Pool: <span className="text-foreground">{JSON.stringify(layer.pool_size)}</span></div>}
                                    </div>
                                </div>
                            ))}
                        </div>
                    </ScrollArea>
                </div>

                <div className="space-y-4">
                    <h3 className="text-lg font-semibold flex items-center gap-2">
                        <Cpu className="h-5 w-5" />
                        Model Summary
                    </h3>
                    <div className="bg-slate-900 text-slate-50 p-4 rounded-lg font-mono text-xs overflow-x-auto h-[400px]">
                        <pre>{arch.model_summary}</pre>
                    </div>
                </div>
            </div>

            <div className="grid grid-cols-3 gap-4">
                <div className="bg-primary/5 border border-primary/20 p-4 rounded-lg text-center">
                    <div className="text-sm text-primary/80 font-medium">Total Parameters</div>
                    <div className="text-2xl font-bold text-primary">{arch.total_params.toLocaleString()}</div>
                </div>
                <div className="bg-muted p-4 rounded-lg text-center">
                    <div className="text-sm text-muted-foreground font-medium">Input Shape</div>
                    <div className="text-2xl font-bold">({arch.input_shape.join(', ')})</div>
                </div>
                <div className="bg-muted p-4 rounded-lg text-center">
                    <div className="text-sm text-muted-foreground font-medium">Classes</div>
                    <div className="text-2xl font-bold">{arch.num_classes}</div>
                </div>
            </div>
        </div>
    )
}
