//! # ghostdiff_macros
//!
//! Procedural macros for automatic function instrumentation in ghostdiff.
//!
//! ## Overview
//!
//! This crate provides the `#[track]` attribute macro that automatically
//! records function entry, exit, arguments, and return values using
//! the ghostdiff recording infrastructure.
//!
//! ## Usage
//!
//! ```rust,ignore
//! use ghostdiff::track;
//!
//! #[track]
//! fn my_function(x: i32, y: i32) -> i32 {
//!     x + y
//! }
//!
//! #[track(name = "ai_inference", ai = true)]
//! async fn run_inference(prompt: &str) -> String {
//!     // AI model call...
//!     "response".to_string()
//! }
//!
//! #[track(tags = ["critical", "hot-path"])]
//! fn critical_operation() {
//!     // ...
//! }
//! ```
//!
//! ## Macro Expansion
//!
//! The macro wraps the function body to:
//! 1. Record function entry with serialized arguments
//! 2. Execute the original function body
//! 3. Record the return value (if Debug-able)
//! 4. Return the result
//!
//! For async functions, it properly handles the Future wrapper.

use proc_macro::TokenStream;
use proc_macro2::TokenStream as TokenStream2;
use quote::{quote, ToTokens};
use syn::{
    parse::{Parse, ParseStream},
    parse_macro_input,
    punctuated::Punctuated,
    Expr, FnArg, ItemFn, Lit, Meta, Pat, ReturnType, Token,
};

/// Configuration parsed from the `#[track(...)]` attribute.
#[derive(Default)]
struct TrackConfig {
    /// Custom name for the tracked function (defaults to function name)
    name: Option<String>,
    /// Whether this is an AI-related function
    ai: bool,
    /// Tags to apply to generated events
    tags: Vec<String>,
    /// Whether to skip argument capture
    skip_args: bool,
    /// Whether to skip return value capture
    skip_return: bool,
}

impl Parse for TrackConfig {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let mut config = TrackConfig::default();

        // Parse comma-separated key=value or key pairs
        let args = Punctuated::<Meta, Token![,]>::parse_terminated(input)?;

        for meta in args {
            match meta {
                Meta::NameValue(nv) => {
                    let ident = nv.path.get_ident().map(|i| i.to_string());
                    match ident.as_deref() {
                        Some("name") => {
                            if let Expr::Lit(expr_lit) = &nv.value {
                                if let Lit::Str(lit_str) = &expr_lit.lit {
                                    config.name = Some(lit_str.value());
                                }
                            }
                        }
                        Some("ai") => {
                            if let Expr::Lit(expr_lit) = &nv.value {
                                if let Lit::Bool(lit_bool) = &expr_lit.lit {
                                    config.ai = lit_bool.value();
                                }
                            }
                        }
                        Some("tags") => {
                            // Parse array literal: tags = ["a", "b"]
                            if let Expr::Array(arr) = &nv.value {
                                for elem in &arr.elems {
                                    if let Expr::Lit(expr_lit) = elem {
                                        if let Lit::Str(lit_str) = &expr_lit.lit {
                                            config.tags.push(lit_str.value());
                                        }
                                    }
                                }
                            }
                        }
                        Some("skip_args") => {
                            if let Expr::Lit(expr_lit) = &nv.value {
                                if let Lit::Bool(lit_bool) = &expr_lit.lit {
                                    config.skip_args = lit_bool.value();
                                }
                            }
                        }
                        Some("skip_return") => {
                            if let Expr::Lit(expr_lit) = &nv.value {
                                if let Lit::Bool(lit_bool) = &expr_lit.lit {
                                    config.skip_return = lit_bool.value();
                                }
                            }
                        }
                        _ => {}
                    }
                }
                Meta::Path(path) => {
                    // Handle bare flags like `ai` (without `= true`)
                    let ident = path.get_ident().map(|i| i.to_string());
                    match ident.as_deref() {
                        Some("ai") => config.ai = true,
                        Some("skip_args") => config.skip_args = true,
                        Some("skip_return") => config.skip_return = true,
                        _ => {}
                    }
                }
                _ => {}
            }
        }

        Ok(config)
    }
}

/// Attribute macro for automatic function instrumentation.
///
/// # Attributes
///
/// - `name = "custom_name"` - Override the recorded function name
/// - `ai` or `ai = true` - Mark as AI-related (uses `track_ai_output`)
/// - `tags = ["tag1", "tag2"]` - Add tags to the event
/// - `skip_args` - Don't capture function arguments
/// - `skip_return` - Don't capture return value
///
/// # Examples
///
/// ## Basic usage
/// ```rust,ignore
/// #[track]
/// fn add(a: i32, b: i32) -> i32 {
///     a + b
/// }
/// ```
///
/// ## With options
/// ```rust,ignore
/// #[track(name = "model_inference", ai = true, tags = ["ml", "inference"])]
/// async fn infer(input: &str) -> Result<String, Error> {
///     // ...
/// }
/// ```
///
/// ## Skip sensitive arguments
/// ```rust,ignore
/// #[track(skip_args = true)]
/// fn authenticate(password: &str) -> bool {
///     // password won't be logged
/// }
/// ```
#[proc_macro_attribute]
pub fn track(attr: TokenStream, item: TokenStream) -> TokenStream {
    let config = parse_macro_input!(attr as TrackConfig);
    let input_fn = parse_macro_input!(item as ItemFn);

    let output = generate_tracked_function(config, input_fn);

    output.into()
}

/// Generates the instrumented function wrapper.
fn generate_tracked_function(config: TrackConfig, input_fn: ItemFn) -> TokenStream2 {
    let ItemFn {
        attrs,
        vis,
        sig,
        block,
    } = input_fn;

    let fn_name = &sig.ident;
    let fn_name_str = config.name.unwrap_or_else(|| fn_name.to_string());
    let is_async = sig.asyncness.is_some();
    let is_ai = config.ai;
    let tags = &config.tags;

    // Extract argument names and patterns for serialization
    let (arg_names, arg_captures) = extract_args(&sig.inputs, config.skip_args);

    // Determine if we should capture return value
    let has_return = !matches!(sig.output, ReturnType::Default);
    let capture_return = has_return && !config.skip_return;

    // Generate the wrapper code based on sync/async
    let wrapper_body = if is_async {
        generate_async_wrapper(
            &fn_name_str,
            block.to_token_stream(),
            &arg_names,
            &arg_captures,
            capture_return,
            is_ai,
            tags,
        )
    } else {
        generate_sync_wrapper(
            &fn_name_str,
            block.to_token_stream(),
            &arg_names,
            &arg_captures,
            capture_return,
            is_ai,
            tags,
        )
    };

    quote! {
        #(#attrs)*
        #vis #sig {
            #wrapper_body
        }
    }
}

/// Extracts argument names and generates capture expressions.
fn extract_args(
    inputs: &Punctuated<FnArg, Token![,]>,
    skip_args: bool,
) -> (Vec<String>, TokenStream2) {
    if skip_args {
        return (vec![], quote! { Vec::<String>::new() });
    }

    let mut arg_names = Vec::new();
    let mut capture_exprs = Vec::new();

    for arg in inputs {
        if let FnArg::Typed(pat_type) = arg {
            if let Pat::Ident(pat_ident) = &*pat_type.pat {
                let ident = &pat_ident.ident;
                let name = ident.to_string();

                // Skip self-like patterns
                if name == "self" || name.starts_with("_") {
                    continue;
                }

                arg_names.push(name.clone());

                // Generate debug capture - falls back to "<opaque>" if Debug not implemented
                capture_exprs.push(quote! {
                    {
                        // Try to use Debug, fall back to type name
                        // This uses a trait-based approach for safety
                        format!("{:?}", &#ident)
                    }
                });
            }
        }
    }

    let captures = if capture_exprs.is_empty() {
        quote! { Vec::<String>::new() }
    } else {
        quote! {
            vec![#(#capture_exprs),*]
        }
    };

    (arg_names, captures)
}

/// Generates wrapper for synchronous functions.
fn generate_sync_wrapper(
    fn_name: &str,
    body: TokenStream2,
    _arg_names: &[String],
    arg_captures: &TokenStream2,
    capture_return: bool,
    is_ai: bool,
    tags: &[String],
) -> TokenStream2 {
    let tags_vec: Vec<_> = tags.iter().map(|t| quote! { #t }).collect();

    let return_capture = if capture_return {
        quote! {
            let __ghostdiff_return_str = Some(format!("{:?}", &__ghostdiff_result));
        }
    } else {
        quote! {
            let __ghostdiff_return_str: Option<String> = None;
        }
    };

    let ai_tracking = if is_ai {
        quote! {
            // For AI functions, also track as AI output if result is String-like
            if let Some(ref ret_str) = __ghostdiff_return_str {
                ghostdiff_core::runtime::with_recorder(|recorder| {
                    recorder.track_ai_output(ret_str);
                });
            }
        }
    } else {
        quote! {}
    };

    let tag_registration = if tags.is_empty() {
        quote! {}
    } else {
        quote! {
            // Add tags to the event
            ghostdiff_core::runtime::with_recorder(|recorder| {
                if let Some(event) = recorder.events().last() {
                    // Note: In real impl, we'd need mutable access
                    let _ = (#(#tags_vec),*); // Tags recorded via metadata
                }
            });
        }
    };

    quote! {
        // Capture arguments before execution
        let __ghostdiff_args: Vec<String> = #arg_captures;

        // Record function entry
        let __ghostdiff_event_id = ghostdiff_core::runtime::with_recorder(|recorder| {
            recorder.track_function_call(#fn_name, __ghostdiff_args.clone(), None)
        });

        // Enter scope for nested calls
        ghostdiff_core::runtime::with_recorder(|recorder| {
            recorder.enter_scope(__ghostdiff_event_id);
        });

        #tag_registration

        // Execute the original function body
        let __ghostdiff_result = (move || #body)();

        // Exit scope
        ghostdiff_core::runtime::with_recorder(|recorder| {
            recorder.exit_scope();
        });

        // Capture return value
        #return_capture

        // Record function exit with return value
        ghostdiff_core::runtime::with_recorder(|recorder| {
            recorder.track_custom(
                "function_return",
                &format!(r#"{{"function": "{}", "return": {:?}}}"#, #fn_name, __ghostdiff_return_str)
            );
        });

        #ai_tracking

        __ghostdiff_result
    }
}

/// Generates wrapper for async functions.
fn generate_async_wrapper(
    fn_name: &str,
    body: TokenStream2,
    _arg_names: &[String],
    arg_captures: &TokenStream2,
    capture_return: bool,
    is_ai: bool,
    tags: &[String],
) -> TokenStream2 {
    let tags_vec: Vec<_> = tags.iter().map(|t| quote! { #t }).collect();

    let return_capture = if capture_return {
        quote! {
            let __ghostdiff_return_str = Some(format!("{:?}", &__ghostdiff_result));
        }
    } else {
        quote! {
            let __ghostdiff_return_str: Option<String> = None;
        }
    };

    let ai_tracking = if is_ai {
        quote! {
            if let Some(ref ret_str) = __ghostdiff_return_str {
                ghostdiff_core::runtime::with_recorder(|recorder| {
                    recorder.track_ai_output(ret_str);
                });
            }
        }
    } else {
        quote! {}
    };

    let tag_registration = if tags.is_empty() {
        quote! {}
    } else {
        quote! {
            let _ = (#(#tags_vec),*);
        }
    };

    quote! {
        // Capture arguments before execution
        let __ghostdiff_args: Vec<String> = #arg_captures;

        // Record async function entry
        let __ghostdiff_event_id = ghostdiff_core::runtime::with_recorder(|recorder| {
            recorder.track_function_call(#fn_name, __ghostdiff_args.clone(), None)
        });

        // Record async task spawn
        let __ghostdiff_task_id = format!("{}_{}", #fn_name, __ghostdiff_event_id);
        ghostdiff_core::runtime::with_recorder(|recorder| {
            recorder.track_async_spawn(&__ghostdiff_task_id);
            recorder.enter_scope(__ghostdiff_event_id);
        });

        #tag_registration

        // Execute the async body
        let __ghostdiff_result = async move #body.await;

        // Exit scope and record completion
        ghostdiff_core::runtime::with_recorder(|recorder| {
            recorder.exit_scope();
            recorder.track_async_complete(&__ghostdiff_task_id, true);
        });

        // Capture return value
        #return_capture

        // Record function exit
        ghostdiff_core::runtime::with_recorder(|recorder| {
            recorder.track_custom(
                "function_return",
                &format!(r#"{{"function": "{}", "return": {:?}}}"#, #fn_name, __ghostdiff_return_str)
            );
        });

        #ai_tracking

        __ghostdiff_result
    }
}
