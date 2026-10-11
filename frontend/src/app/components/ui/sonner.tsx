import { Toaster as Sonner, type ToasterProps } from "sonner";

// The app ships a light theme only, so the toaster is pinned to it.
const Toaster = (props: ToasterProps) => (
  <Sonner theme="light" className="toaster group" {...props} />
);

export { Toaster };
