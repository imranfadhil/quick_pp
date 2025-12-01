<script lang="ts">
  export let variant: 'ghost' | 'primary' | 'danger' = 'ghost';
  export let size: 'sm' | 'md' = 'md';
  export let disabled: boolean = false;
  export let title: string | undefined = undefined;
  export let type: 'button' | 'submit' | 'reset' = 'button';
  export let onclick: ((...args: any[]) => any) | undefined = undefined;
  export let ariaLabel: string | undefined = undefined;
  export let className: string = '';

  const variantClass = variant === 'primary' ? 'btn-primary' : variant === 'danger' ? 'btn-danger' : 'btn-ghost';
  const sizeClass = size === 'sm' ? 'btn-sm' : '';
</script>

<button
  {type}
  {title}
  aria-label={ariaLabel}
  class={`btn ${variantClass} ${sizeClass} ${className}`}
  {disabled}
  onclick={onclick}
>
  <slot />
</button>

<style>
  /* Button styles centralized here for reuse */
  .btn {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    gap: .4rem;
    padding: .35rem .6rem;
    border-radius: 6px;
    border: 1px solid transparent;
    font-size: .875rem;
    line-height: 1;
    cursor: pointer;
    user-select: none;
    transition: background .12s ease, color .12s ease, box-shadow .12s ease;
    background: transparent;
    color: #0f172a;
  }

  .btn:disabled { opacity: .55; cursor: not-allowed }

  .btn-primary { background: #1366d6; color: #fff; border-color: rgba(2,6,23,0.08) }
  .btn-primary:hover:not(:disabled) { background: #0f57c2 }

  .btn-ghost { background: transparent; color: #0f172a; border: 1px solid rgba(15,23,42,0.06) }
  .btn-ghost:hover:not(:disabled) { background: rgba(2,6,23,0.04) }

  .btn-danger { background: #d14343; color: #fff }

  .btn-sm { font-size: .8rem; padding: .25rem .5rem; border-radius: 5px }

  .btn:focus { outline: 3px solid rgba(19,102,214,0.18); outline-offset: 2px }
</style>
