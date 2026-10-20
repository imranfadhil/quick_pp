<script lang="ts">
	import { onMount } from 'svelte';
	import { goto } from '$app/navigation';
	import logo from '$lib/assets/logo.png';
	import { getJson, postJson } from '$lib/api/client';
	import { authState, type AuthUser, type MfaSetupPayload } from '$lib/state/auth.svelte';

	type AuthStep =
		| 'SIGN_IN'
		| 'NEW_PASSWORD_REQUIRED'
		| 'MFA_SETUP'
		| 'SOFTWARE_TOKEN_MFA';

	let step = $state<AuthStep>('SIGN_IN');
	let isLoading = $state(false);
	let errorMsg = $state<string | null>(null);
	let infoMsg = $state<string | null>(null);

	// Step 1: Sign-in fields
	let username = $state('');
	let password = $state('');
	let inviteToken = $state<string | null>(null);
	let showPassword = $state(false);

	// Bootstrap status (first-run admin onboarding helper)
	let isFirstRun = $state(false);
	let bootstrapUsername = $state<string | null>(null);

	// Challenge state
	let challengeToken = $state('');
	let pendingUser = $state<AuthUser | null>(null);

	// Step 2: New Password Required fields
	let fullName = $state('');
	let newPassword = $state('');
	let confirmPassword = $state('');
	let showNewPassword = $state(false);

	// Step 3 & 4: TOTP MFA fields
	let mfaSetup = $state<MfaSetupPayload | null>(null);
	let totpCode = $state('');
	let copiedSecret = $state(false);

	// Real-time Cognito password policy checks
	let pwdChecks = $derived({
		length: newPassword.length >= 8,
		upper: /[A-Z]/.test(newPassword),
		lower: /[a-z]/.test(newPassword),
		digit: /\d/.test(newPassword),
		symbol: /[^A-Za-z0-9]/.test(newPassword),
		match: newPassword.length > 0 && newPassword === confirmPassword
	});

	let isNewPasswordValid = $derived(
		pwdChecks.length &&
			pwdChecks.upper &&
			pwdChecks.lower &&
			pwdChecks.digit &&
			pwdChecks.symbol &&
			pwdChecks.match
	);

	let formattedMfaSecret = $derived.by(() => {
		const raw = mfaSetup?.secret ?? '';
		return raw.replace(/(.{4})/g, '$1 ').trim();
	});

	onMount(async () => {
		const params = new URLSearchParams(window.location.search);
		const urlUser = params.get('username');
		const urlInvite = params.get('invite');
		if (urlUser) username = urlUser;
		if (urlInvite) {
			inviteToken = urlInvite;
			infoMsg = 'Administrator invitation link detected. Click continue to set up your account.';
		}

		// If already authenticated and no invite link is being claimed, redirect to workbench
		if (!urlInvite) {
			const ok = await authState.initialize();
			if (ok) {
				goto('/');
				return;
			}
		}

		try {
			const boot = await getJson<{
				is_first_run: boolean;
				admin_username: string | null;
			}>('/quick_pp/auth/bootstrap-status');
			if (boot?.is_first_run) {
				isFirstRun = true;
				bootstrapUsername = boot.admin_username || 'admin';
			}
		} catch {
			// Ignore bootstrap check error
		}
	});

	function fillBootstrapCredentials() {
		username = bootstrapUsername || 'admin';
		password = 'QuickPP@Admin2026!';
		inviteToken = null;
		errorMsg = null;
	}

	async function handleInitiateAuth(e: Event) {
		e.preventDefault();
		errorMsg = null;
		infoMsg = null;
		isLoading = true;

		try {
			const res = await postJson<{
				status: string;
				challenge_name?: AuthStep;
				challenge_token?: string;
				message?: string;
				mfa_setup?: MfaSetupPayload;
				user?: AuthUser;
				access_token?: string;
			}>('/quick_pp/auth/login', {
				username: username.trim(),
				password: password || undefined,
				invite_token: inviteToken || undefined
			});

			if (res.status === 'AUTHENTICATED' && res.access_token && res.user) {
				authState.setAuthenticated(res.access_token, res.user);
				await goto('/');
				return;
			}

			if (res.status === 'CHALLENGE' && res.challenge_name && res.challenge_token) {
				challengeToken = res.challenge_token;
				pendingUser = res.user ?? null;
				if (pendingUser?.full_name) {
					fullName = pendingUser.full_name;
				}
				if (res.mfa_setup) {
					mfaSetup = res.mfa_setup;
				}
				step = res.challenge_name;
				infoMsg = res.message ?? null;
				totpCode = '';
			}
		} catch (err: any) {
			errorMsg = err?.message || 'Sign in failed. Verify your credentials.';
		} finally {
			isLoading = false;
		}
	}

	async function handleNewPasswordSubmit(e: Event) {
		e.preventDefault();
		if (!isNewPasswordValid) return;
		errorMsg = null;
		infoMsg = null;
		isLoading = true;

		try {
			const res = await postJson<{
				status: string;
				challenge_name: AuthStep;
				challenge_token: string;
				message?: string;
				mfa_setup?: MfaSetupPayload;
				user?: AuthUser;
			}>('/quick_pp/auth/challenge/new-password', {
				challenge_token: challengeToken,
				new_password: newPassword,
				full_name: fullName.trim() || undefined
			});

			challengeToken = res.challenge_token;
			pendingUser = res.user ?? pendingUser;
			if (res.mfa_setup) {
				mfaSetup = res.mfa_setup;
			}
			step = res.challenge_name;
			infoMsg = res.message ?? null;
			totpCode = '';
		} catch (err: any) {
			errorMsg = err?.message || 'Failed to update password.';
		} finally {
			isLoading = false;
		}
	}

	async function handleMfaSubmit(e: Event) {
		e.preventDefault();
		const cleaned = totpCode.replace(/\s+/g, '');
		if (cleaned.length !== 6) {
			errorMsg = 'Please enter a 6-digit verification code.';
			return;
		}
		errorMsg = null;
		infoMsg = null;
		isLoading = true;

		const endpoint =
			step === 'MFA_SETUP'
				? '/quick_pp/auth/challenge/mfa-verify-setup'
				: '/quick_pp/auth/challenge/mfa-verify';

		try {
			const res = await postJson<{
				status: string;
				access_token: string;
				user: AuthUser;
			}>(endpoint, {
				challenge_token: challengeToken,
				totp_code: cleaned
			});

			if (res.status === 'AUTHENTICATED' && res.access_token && res.user) {
				authState.setAuthenticated(res.access_token, res.user);
				await goto('/');
			}
		} catch (err: any) {
			errorMsg = err?.message || 'Invalid 6-digit verification code.';
		} finally {
			isLoading = false;
		}
	}

	function handleTotpInput(e: Event) {
		const target = e.currentTarget as HTMLInputElement;
		totpCode = target.value.replace(/\D/g, '').slice(0, 6);
	}

	async function copySecretKey() {
		if (!mfaSetup?.secret) return;
		try {
			await navigator.clipboard.writeText(mfaSetup.secret);
			copiedSecret = true;
			setTimeout(() => (copiedSecret = false), 2000);
		} catch {
			// Ignore clipboard error
		}
	}

	function resetToSignIn() {
		step = 'SIGN_IN';
		challengeToken = '';
		errorMsg = null;
		infoMsg = null;
		totpCode = '';
		newPassword = '';
		confirmPassword = '';
	}
</script>

<div
	class="relative flex min-h-screen w-screen flex-col items-center justify-center bg-[var(--bg-app)] px-4 py-8 text-[var(--text-primary)] select-none"
>
	<!-- Subtle engineering grid background -->
	<div
		class="pointer-events-none absolute inset-0 opacity-45"
		style="background-image: linear-gradient(to right, #E2E8F0 1px, transparent 1px), linear-gradient(to bottom, #E2E8F0 1px, transparent 1px); background-size: 32px 32px;"
	></div>

	<div class="relative z-10 w-full max-w-[420px]">
		<!-- Top Brand Header -->
		<div class="mb-4 flex items-center justify-between">
			<div class="flex items-center gap-2.5">
				<div
					class="flex h-8 w-8 items-center justify-center"
				>
				<img
					src={logo}
					alt="quick_pp logo"
					class="h-8 w-8 shrink-0 object-contain"
				/>
				</div>
				<div>
					<div class="flex items-center gap-2">
						<span class="text-sm font-bold tracking-tight text-[var(--text-primary)]">
							quick_pp
						</span>
						<span
							class="rounded-xs border border-blue-200 bg-blue-50 px-1.5 py-0.5 text-[10px] font-semibold tracking-wide text-blue-700 uppercase"
						>
							Identity & UAM
						</span>
					</div>
					<p class="text-[11px] text-[var(--text-muted)]">
						Subsurface Petrophysical Evaluation Workbench
					</p>
				</div>
			</div>
		</div>

		<!-- Main Card -->
		<div
			class="rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-surface)] shadow-sm"
		>
			<!-- Step Progress Bar -->
			<div
				class="flex items-center justify-between border-b border-[var(--border-hairline)] bg-[var(--bg-subtle)] px-4 py-2 text-[11px]"
			>
				<div class="flex items-center gap-1.5">
					<span
						class="flex h-4 w-4 items-center justify-center rounded-full text-[10px] font-bold {step ===
						'SIGN_IN'
							? 'bg-[var(--accent-primary)] text-white'
							: 'bg-emerald-600 text-white'}"
					>
						{step === 'SIGN_IN' ? '1' : '✓'}
					</span>
					<span
						class={step === 'SIGN_IN'
							? 'font-semibold text-[var(--text-primary)]'
							: 'text-[var(--text-muted)]'}
					>
						Credentials
					</span>
				</div>

				<span class="text-[var(--border-hairline)]">―</span>

				<div class="flex items-center gap-1.5">
					<span
						class="flex h-4 w-4 items-center justify-center rounded-full text-[10px] font-bold {step ===
						'NEW_PASSWORD_REQUIRED'
							? 'bg-[var(--accent-primary)] text-white'
							: step === 'MFA_SETUP' || step === 'SOFTWARE_TOKEN_MFA'
								? 'bg-emerald-600 text-white'
								: 'border border-[var(--border-hairline)] bg-white text-[var(--text-muted)]'}"
					>
						{step === 'MFA_SETUP' || step === 'SOFTWARE_TOKEN_MFA' ? '✓' : '2'}
					</span>
					<span
						class={step === 'NEW_PASSWORD_REQUIRED'
							? 'font-semibold text-[var(--text-primary)]'
							: 'text-[var(--text-muted)]'}
					>
						Password Policy
					</span>
				</div>

				<span class="text-[var(--border-hairline)]">―</span>

				<div class="flex items-center gap-1.5">
					<span
						class="flex h-4 w-4 items-center justify-center rounded-full text-[10px] font-bold {step ===
							'MFA_SETUP' || step === 'SOFTWARE_TOKEN_MFA'
							? 'bg-[var(--accent-primary)] text-white'
							: 'border border-[var(--border-hairline)] bg-white text-[var(--text-muted)]'}"
					>
						3
					</span>
					<span
						class={step === 'MFA_SETUP' || step === 'SOFTWARE_TOKEN_MFA'
							? 'font-semibold text-[var(--text-primary)]'
							: 'text-[var(--text-muted)]'}
					>
						TOTP MFA
					</span>
				</div>
			</div>

			<div class="p-5">
				{#if errorMsg}
					<div
						class="mb-3.5 rounded-sm border border-red-200 bg-red-50 px-3 py-2 text-xs text-red-700"
					>
						<div class="font-semibold">Authentication Alert</div>
						<div class="mt-0.5 text-[11px]">{errorMsg}</div>
					</div>
				{/if}

				{#if infoMsg}
					<div
						class="mb-3.5 rounded-sm border border-blue-200 bg-blue-50 px-3 py-2 text-xs text-blue-800"
					>
						{infoMsg}
					</div>
				{/if}

				<!-- ========================================================= -->
				<!-- STEP 1: SIGN IN (Credentials or Invitation Link)          -->
				<!-- ========================================================= -->
				{#if step === 'SIGN_IN'}
					<div class="mb-3.5">
						<h1 class="text-sm font-bold text-[var(--text-primary)]">
							Sign In to Workbench
						</h1>
						<p class="mt-0.5 text-[11px] text-[var(--text-muted)]">
							Access is restricted to invited users with Multi-Factor Authentication.
						</p>
					</div>

					{#if isFirstRun && !inviteToken}
						<div
							class="mb-3.5 rounded-sm border border-amber-300 bg-amber-50/90 p-2.5 text-[11px] text-amber-900"
						>
							<div class="flex items-center justify-between">
								<span class="font-semibold">First-Time Administrator Onboarding</span>
								<button
									type="button"
									onclick={fillBootstrapCredentials}
									class="rounded-xs bg-amber-700 px-2 py-0.5 text-[10px] font-semibold text-white hover:bg-amber-800"
								>
									Auto-Fill Admin
								</button>
							</div>
							<p class="mt-1 text-[10px] leading-relaxed text-amber-800">
								Default bootstrap account <span class="mono-num font-semibold">{bootstrapUsername}</span> is ready. Sign in with temporary password <span class="mono-num font-semibold">QuickPP@Admin2026!</span> to set your permanent password and enroll MFA.
							</p>
						</div>
					{/if}

					{#if inviteToken}
						<div
							class="mb-3.5 flex items-center justify-between rounded-sm border border-emerald-200 bg-emerald-50 px-3 py-2 text-[11px] text-emerald-900"
						>
							<div>
								<div class="font-semibold">Invitation Token Active</div>
								<div class="text-[10px] text-emerald-700">
									Account: <span class="mono-num font-semibold">{username || 'Invited User'}</span>
								</div>
							</div>
							<button
								type="button"
								onclick={() => {
									inviteToken = null;
									infoMsg = null;
								}}
								class="text-[10px] text-emerald-700 underline hover:text-emerald-900"
							>
								Use password instead
							</button>
						</div>
					{/if}

					<form onsubmit={handleInitiateAuth} class="space-y-3">
						<label class="block">
							<span class="mb-1 block text-xs font-medium text-[var(--text-secondary)]">
								Username or Email
							</span>
							<input
								type="text"
								bind:value={username}
								required
								autocomplete="username"
								placeholder="e.g. petrophysicist or name@company.com"
								class="h-8 w-full rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-surface)] px-2.5 text-xs text-[var(--text-primary)] focus:border-[var(--border-focus)] focus:outline-none"
							/>
						</label>

						{#if !inviteToken}
							<label class="block">
								<div class="mb-1 flex items-center justify-between">
									<span class="text-xs font-medium text-[var(--text-secondary)]">
										Password / Temporary Password
									</span>
									<button
										type="button"
										onclick={() => (showPassword = !showPassword)}
										class="text-[10px] text-[var(--text-muted)] hover:text-[var(--text-primary)]"
									>
										{showPassword ? 'Hide' : 'Show'}
									</button>
								</div>
								<input
									type={showPassword ? 'text' : 'password'}
									bind:value={password}
									required={!inviteToken}
									autocomplete="current-password"
									placeholder="Enter your password"
									class="h-8 w-full rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-surface)] px-2.5 text-xs text-[var(--text-primary)] focus:border-[var(--border-focus)] focus:outline-none"
								/>
							</label>
						{/if}

						<button
							type="submit"
							disabled={isLoading}
							class="mt-1 flex h-8 w-full items-center justify-center rounded-sm bg-[var(--accent-primary)] text-xs font-semibold text-white transition-colors hover:bg-blue-700 disabled:opacity-50"
						>
							{#if isLoading}
								Verifying Credentials...
							{:else if inviteToken}
								Accept Invitation & Set Password →
							{:else}
								Continue to Verification →
							{/if}
						</button>
					</form>

				<!-- ========================================================= -->
				<!-- STEP 2: NEW_PASSWORD_REQUIRED (Force Change Password)     -->
				<!-- ========================================================= -->
				{:else if step === 'NEW_PASSWORD_REQUIRED'}
					<div class="mb-3">
						<div class="flex items-center justify-between">
							<h1 class="text-sm font-bold text-[var(--text-primary)]">
								Set Permanent Password
							</h1>
							<span
								class="rounded-xs bg-amber-100 px-1.5 py-0.5 text-[10px] font-semibold text-amber-800"
							>
								FORCE_CHANGE_PASSWORD
							</span>
						</div>
						<p class="mt-0.5 text-[11px] text-[var(--text-muted)]">
							Account <span class="mono-num font-semibold text-[var(--text-primary)]">{pendingUser?.username}</span> requires a permanent password before enrolling MFA.
						</p>
					</div>

					<form onsubmit={handleNewPasswordSubmit} class="space-y-3">
						<label class="block">
							<span class="mb-1 block text-xs font-medium text-[var(--text-secondary)]">
								Full Name / Display Name
							</span>
							<input
								type="text"
								bind:value={fullName}
								placeholder="e.g. Dr. Imran Fadhil"
								class="h-8 w-full rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-surface)] px-2.5 text-xs"
							/>
						</label>

						<label class="block">
							<div class="mb-1 flex items-center justify-between">
								<span class="text-xs font-medium text-[var(--text-secondary)]">
									New Permanent Password
								</span>
								<button
									type="button"
									onclick={() => (showNewPassword = !showNewPassword)}
									class="text-[10px] text-[var(--text-muted)] hover:text-[var(--text-primary)]"
								>
									{showNewPassword ? 'Hide' : 'Show'}
								</button>
							</div>
							<input
								type={showNewPassword ? 'text' : 'password'}
								bind:value={newPassword}
								required
								autocomplete="new-password"
								placeholder="Create a strong password"
								class="h-8 w-full rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-surface)] px-2.5 text-xs"
							/>
						</label>

						<label class="block">
							<span class="mb-1 block text-xs font-medium text-[var(--text-secondary)]">
								Confirm New Password
							</span>
							<input
								type={showNewPassword ? 'text' : 'password'}
								bind:value={confirmPassword}
								required
								autocomplete="new-password"
								placeholder="Re-enter new password"
								class="h-8 w-full rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-surface)] px-2.5 text-xs"
							/>
						</label>

						<!-- Cognito Password Policy Checklist -->
						<div
							class="rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-subtle)] p-2.5 text-[11px]"
						>
							<div class="mb-1.5 text-[10px] font-semibold tracking-wider text-[var(--text-muted)] uppercase">
								Password Security Policy
							</div>
							<div class="grid grid-cols-2 gap-1 text-[11px]">
								<div class={pwdChecks.length ? 'font-medium text-emerald-700' : 'text-[var(--text-muted)]'}>
									{pwdChecks.length ? '✓' : '○'} Min 8 characters
								</div>
								<div class={pwdChecks.upper ? 'font-medium text-emerald-700' : 'text-[var(--text-muted)]'}>
									{pwdChecks.upper ? '✓' : '○'} Uppercase (A–Z)
								</div>
								<div class={pwdChecks.lower ? 'font-medium text-emerald-700' : 'text-[var(--text-muted)]'}>
									{pwdChecks.lower ? '✓' : '○'} Lowercase (a–z)
								</div>
								<div class={pwdChecks.digit ? 'font-medium text-emerald-700' : 'text-[var(--text-muted)]'}>
									{pwdChecks.digit ? '✓' : '○'} Number (0–9)
								</div>
								<div class={pwdChecks.symbol ? 'font-medium text-emerald-700' : 'text-[var(--text-muted)]'}>
									{pwdChecks.symbol ? '✓' : '○'} Special symbol (!@#$)
								</div>
								<div class={pwdChecks.match ? 'font-medium text-emerald-700' : 'text-[var(--text-muted)]'}>
									{pwdChecks.match ? '✓' : '○'} Passwords match
								</div>
							</div>
						</div>

						<div class="flex gap-2 pt-1">
							<button
								type="button"
								onclick={resetToSignIn}
								class="h-8 rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-surface)] px-3 text-xs font-medium text-[var(--text-secondary)] hover:bg-[var(--bg-subtle)]"
							>
								Back
							</button>
							<button
								type="submit"
								disabled={!isNewPasswordValid || isLoading}
								class="flex-1 h-8 rounded-sm bg-[var(--accent-primary)] text-xs font-semibold text-white hover:bg-blue-700 disabled:opacity-50"
							>
								{isLoading ? 'Saving Password...' : 'Save Password & Configure MFA →'}
							</button>
						</div>
					</form>

				<!-- ========================================================= -->
				<!-- STEP 3: MFA_SETUP (QR Code Authenticator Provisioning)    -->
				<!-- ========================================================= -->
				{:else if step === 'MFA_SETUP'}
					<div class="mb-3">
						<div class="flex items-center justify-between">
							<h1 class="text-sm font-bold text-[var(--text-primary)]">
								Enroll Authenticator MFA
							</h1>
							<span
								class="rounded-xs bg-blue-100 px-1.5 py-0.5 text-[10px] font-semibold text-blue-800"
							>
								TOTP SETUP
							</span>
						</div>
						<p class="mt-0.5 text-[11px] text-[var(--text-muted)]">
							Scan the QR code below using Google Authenticator, Microsoft Authenticator, Authy, or 1Password.
						</p>
					</div>

					{#if mfaSetup}
						<div
							class="mb-3 flex flex-col items-center rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-subtle)] p-3"
						>
							<div
								class="rounded-sm border border-[var(--border-hairline)] bg-white p-2 shadow-2xs"
							>
								<img
									src={mfaSetup.qr_data_uri}
									alt="TOTP Authenticator QR Code"
									class="h-40 w-40"
								/>
							</div>

							<div class="mt-2.5 w-full">
								<div class="mb-1 flex items-center justify-between text-[10px] text-[var(--text-muted)]">
									<span>Manual Setup Secret Key (Base32)</span>
									<button
										type="button"
										onclick={copySecretKey}
										class="font-semibold text-[var(--accent-primary)] hover:underline"
									>
										{copiedSecret ? '✓ Copied' : 'Copy Key'}
									</button>
								</div>
								<div
									class="mono-num w-full rounded-xs border border-[var(--border-hairline)] bg-white px-2 py-1 text-center text-[11px] font-semibold tracking-wider text-[var(--text-primary)] select-all"
								>
									{formattedMfaSecret}
								</div>
							</div>
						</div>
					{/if}

					<form onsubmit={handleMfaSubmit} class="space-y-3">
						<label class="block">
							<span class="mb-1 block text-xs font-medium text-[var(--text-secondary)]">
								Enter 6-Digit Authenticator Code to Activate
							</span>
							<input
								type="text"
								inputmode="numeric"
								autocomplete="one-time-code"
								maxlength="6"
								value={totpCode}
								oninput={handleTotpInput}
								placeholder="000000"
								required
								class="mono-num h-9 w-full rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-surface)] px-3 text-center text-base font-bold tracking-[0.35em] text-[var(--text-primary)] focus:border-[var(--border-focus)] focus:outline-none"
							/>
						</label>

						<div class="flex gap-2">
							<button
								type="button"
								onclick={resetToSignIn}
								class="h-8 rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-surface)] px-3 text-xs font-medium text-[var(--text-secondary)] hover:bg-[var(--bg-subtle)]"
							>
								Cancel
							</button>
							<button
								type="submit"
								disabled={totpCode.length !== 6 || isLoading}
								class="flex-1 h-8 rounded-sm bg-[var(--accent-primary)] text-xs font-semibold text-white hover:bg-blue-700 disabled:opacity-50"
							>
								{isLoading ? 'Activating MFA...' : 'Verify & Complete Sign In'}
							</button>
						</div>
					</form>

				<!-- ========================================================= -->
				<!-- STEP 4: SOFTWARE_TOKEN_MFA (Returning User 2FA Prompt)    -->
				<!-- ========================================================= -->
				{:else if step === 'SOFTWARE_TOKEN_MFA'}
					<div class="mb-3.5">
						<div class="flex items-center justify-between">
							<h1 class="text-sm font-bold text-[var(--text-primary)]">
								Multi-Factor Authentication
							</h1>
							<span
								class="rounded-xs bg-emerald-100 px-1.5 py-0.5 text-[10px] font-semibold text-emerald-800"
							>
								2FA VERIFY
							</span>
						</div>
						<p class="mt-0.5 text-[11px] text-[var(--text-muted)]">
							Enter the 6-digit code from your Authenticator app for <span class="mono-num font-semibold text-[var(--text-primary)]">{pendingUser?.username || username}</span>.
						</p>
					</div>

					<form onsubmit={handleMfaSubmit} class="space-y-3.5">
						<label class="block">
							<span class="mb-1 block text-xs font-medium text-[var(--text-secondary)]">
								6-Digit Verification Code
							</span>
							<input
								type="text"
								inputmode="numeric"
								autocomplete="one-time-code"
								maxlength="6"
								value={totpCode}
								oninput={handleTotpInput}
								placeholder="000000"
								required
								class="mono-num h-10 w-full rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-surface)] px-3 text-center text-lg font-bold tracking-[0.4em] text-[var(--text-primary)] focus:border-[var(--border-focus)] focus:outline-none"
							/>
						</label>

						<div class="flex gap-2">
							<button
								type="button"
								onclick={resetToSignIn}
								class="h-8 rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-surface)] px-3 text-xs font-medium text-[var(--text-secondary)] hover:bg-[var(--bg-subtle)]"
							>
								Back
							</button>
							<button
								type="submit"
								disabled={totpCode.length !== 6 || isLoading}
								class="flex-1 h-8 rounded-sm bg-[var(--accent-primary)] text-xs font-semibold text-white hover:bg-blue-700 disabled:opacity-50"
							>
								{isLoading ? 'Verifying Code...' : 'Verify & Open Workbench'}
							</button>
						</div>
					</form>
				{/if}
			</div>

			<!-- Card Footer -->
			<div
				class="flex items-center justify-between border-t border-[var(--border-hairline)] bg-[var(--bg-subtle)] px-4 py-2 text-[10px] text-[var(--text-muted)]"
			>
				<span>Invite-Only Access Control • RFC 6238 TOTP</span>
				<span class="mono-num">PBKDF2-SHA256 / JWT</span>
			</div>
		</div>
	</div>
</div>
