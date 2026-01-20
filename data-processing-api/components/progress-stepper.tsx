"use client"

import { Check } from "lucide-react"
import { cn } from "@/lib/utils"

interface Step {
  title: string
  description?: string
}

interface ProgressStepperProps {
  steps: Step[]
  currentStep: number
}

export function ProgressStepper({ steps, currentStep }: ProgressStepperProps) {
  return (
    <div className="w-full">
      <nav aria-label="Progress">
        <ol className="flex items-center justify-between">
          {steps.map((step, index) => {
            const isCompleted = index < currentStep
            const isCurrent = index === currentStep
            const isUpcoming = index > currentStep

            return (
              <li
                key={index}
                className={cn(
                  "relative flex-1",
                  index !== steps.length - 1 && "pr-8 sm:pr-20"
                )}
              >
                {/* Connector Line */}
                {index !== steps.length - 1 && (
                  <div
                    className={cn(
                      "absolute left-0 top-4 -ml-px mt-0.5 h-0.5 w-full",
                      isCompleted ? "bg-primary" : "bg-muted"
                    )}
                    aria-hidden="true"
                  />
                )}

                {/* Step */}
                <div className="group relative flex flex-col items-center">
                  <span className="flex h-9 items-center" aria-hidden="true">
                    <span
                      className={cn(
                        "relative z-10 flex h-8 w-8 items-center justify-center rounded-full border-2 transition-colors",
                        isCompleted &&
                          "border-primary bg-primary text-primary-foreground",
                        isCurrent &&
                          "border-primary bg-background text-primary",
                        isUpcoming && "border-muted bg-background text-muted-foreground"
                      )}
                    >
                      {isCompleted ? (
                        <Check className="h-5 w-5" />
                      ) : (
                        <span className="text-sm font-semibold">{index + 1}</span>
                      )}
                    </span>
                  </span>
                  <span className="mt-2 flex min-w-0 flex-col items-center">
                    <span
                      className={cn(
                        "text-xs font-medium transition-colors text-center",
                        isCurrent && "text-primary",
                        isCompleted && "text-foreground",
                        isUpcoming && "text-muted-foreground"
                      )}
                    >
                      {step.title}
                    </span>
                    {step.description && (
                      <span className="text-xs text-muted-foreground text-center mt-1 hidden sm:block">
                        {step.description}
                      </span>
                    )}
                  </span>
                </div>
              </li>
            )
          })}
        </ol>
      </nav>
    </div>
  )
}
