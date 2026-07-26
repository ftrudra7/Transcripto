import * as React from "react";
import { Slot } from "@radix-ui/react-slot";
import { cva, type VariantProps } from "class-variance-authority";
import { cn } from "@/lib/utils";
import { motion } from "framer-motion";

const buttonVariants = cva(
  "inline-flex items-center justify-center whitespace-nowrap rounded-full text-sm font-medium ring-offset-background transition-all focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50",
  {
    variants: {
      variant: {
        default:
          "bg-[linear-gradient(135deg,var(--color-primary),var(--color-secondary))] text-white border border-white/20 shadow-[0_4px_15px_var(--color-primary-glow)] hover:shadow-[0_6px_20px_var(--color-primary-glow)] hover:brightness-110",
        destructive:
          "bg-red-500/10 text-red-400 border border-red-500/30 hover:bg-red-500/20 hover:text-white",
        outline:
          "border border-[var(--color-glass-border)] bg-[var(--color-surface)] hover:bg-[var(--color-surface-hover)] text-[var(--color-foreground)]",
        secondary:
          "bg-white/5 text-[var(--color-foreground)] hover:bg-white/10",
        ghost: "hover:bg-white/5 hover:text-white text-[var(--color-muted)]",
        link: "text-[var(--color-primary)] underline-offset-4 hover:underline",
      },
      size: {
        default: "h-10 px-5 py-2",
        sm: "h-9 rounded-full px-4",
        lg: "h-11 rounded-full px-8 text-base",
        icon: "h-10 w-10",
      },
    },
    defaultVariants: {
      variant: "default",
      size: "default",
    },
  }
);

export interface ButtonProps
  extends React.ButtonHTMLAttributes<HTMLButtonElement>,
    VariantProps<typeof buttonVariants> {
  asChild?: boolean;
}

const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant, size, asChild = false, ...props }, ref) => {
    const Comp = asChild ? Slot : "button";
    return (
      <motion.div whileHover={{ y: -1 }} whileTap={{ y: 1 }} className="inline-block">
        <Comp
          className={cn(buttonVariants({ variant, size, className }))}
          ref={ref}
          {...props}
        />
      </motion.div>
    );
  }
);
Button.displayName = "Button";

export { Button, buttonVariants };
