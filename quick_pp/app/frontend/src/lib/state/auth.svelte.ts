import { goto } from '$app/navigation';
import {
	deleteJson,
	getJson,
	getStoredToken,
	patchJson,
	postJson,
	setStoredToken
} from '$lib/api/client';

export interface UserProjectMembership {
	project_id: number;
	project_name: string;
	role: 'viewer' | 'editor' | 'admin';
}

export interface AuthUser {
	user_id: number;
	username: string;
	email: string;
	full_name: string;
	role: 'admin' | 'user';
	status: 'FORCE_CHANGE_PASSWORD' | 'MFA_SETUP_REQUIRED' | 'CONFIRMED' | 'DISABLED';
	is_active: boolean;
	mfa_enabled: boolean;
	has_invite_token?: boolean;
	invited_by?: number | null;
	last_login_at?: string | null;
	created_at?: string;
	project_memberships: UserProjectMembership[];
	demo_mode?: boolean;
}

export interface MfaSetupPayload {
	secret: string;
	otpauth_uri: string;
	qr_data_uri: string;
	issuer: string;
	account_name: string;
}

export interface InvitationDetails {
	username: string;
	email: string;
	temp_password: string;
	invite_token: string;
	invite_url: string;
	email_sent: boolean;
	email_message: string;
}

export interface AuditLogItem {
	log_id: number;
	user_id: number | null;
	username: string;
	action: string;
	table_name: string;
	record_id: string | null;
	timestamp: string;
	changes: Record<string, any>;
}

class AuthState {
	user = $state<AuthUser | null>(null);
	token = $state<string | null>(null);
	isInitializing = $state<boolean>(true);
	uamModalOpen = $state<boolean>(false);
	accountModalOpen = $state<boolean>(false);
	smtpConfigured = $state<boolean>(false);
	smtpHost = $state<string | null>(null);

	isAuthenticated = $derived<boolean>(Boolean(this.user && this.token));
	isAdmin = $derived<boolean>(this.user?.role === 'admin');
	isDemoMode = $derived<boolean>(Boolean(this.user?.demo_mode));

	userInitials = $derived.by(() => {
		if (!this.user) return 'QP';
		const source = (this.user.full_name || this.user.username || 'U').trim();
		const parts = source.split(/\s+/);
		if (parts.length >= 2) {
			return (parts[0][0] + parts[1][0]).toUpperCase();
		}
		return source.slice(0, 2).toUpperCase();
	});

	async initialize(): Promise<boolean> {
		this.isInitializing = true;
		try {
			const stored = getStoredToken();
			this.token = stored;
			const res = await getJson<{ user: AuthUser }>('/quick_pp/auth/me');
			if (res?.user) {
				this.user = res.user;
				if (res.user.demo_mode) {
					setStoredToken(null);
					this.token = 'demo-session';
				} else if (!this.token) {
					this.token = 'cookie-session';
				}
				return true;
			}
			this.clearSession();
			return false;
		} catch {
			this.clearSession();
			return false;
		} finally {
			this.isInitializing = false;
		}
	}

	setAuthenticated(accessToken: string, user: AuthUser) {
		if (user.demo_mode) {
			setStoredToken(null);
			this.token = 'demo-session';
		} else {
			setStoredToken(accessToken);
			this.token = accessToken;
		}
		this.user = user;
	}

	clearSession() {
		setStoredToken(null);
		this.token = null;
		this.user = null;
		this.uamModalOpen = false;
		this.accountModalOpen = false;
	}

	async logout() {
		if (this.isDemoMode) {
			await goto('/');
			return;
		}
		try {
			await postJson('/quick_pp/auth/logout', {});
		} catch {
			// Ignore logout error
		}
		this.clearSession();
		await goto('/login');
	}

	getProjectRole(projectId: string | number | null): 'admin' | 'editor' | 'viewer' {
		if (!this.user) return 'viewer';
		if (this.user.role === 'admin') return 'admin';
		if (!projectId) return 'viewer';
		const pid = Number(projectId);
		const pm = this.user.project_memberships?.find((m) => m.project_id === pid);
		return pm?.role ?? 'viewer';
	}

	canEditProject(projectId: string | number | null): boolean {
		const role = this.getProjectRole(projectId);
		return role === 'admin' || role === 'editor';
	}

	// --- Admin UAM API Helpers ---

	async listUsers(): Promise<AuthUser[]> {
		const res = await getJson<{
			users: AuthUser[];
			smtp_configured?: boolean;
			smtp_host?: string | null;
		}>('/quick_pp/uam/users');
		this.smtpConfigured = Boolean(res.smtp_configured);
		this.smtpHost = res.smtp_host ?? null;
		return res.users || [];
	}

	async inviteUser(payload: {
		username: string;
		email: string;
		full_name?: string;
		role: 'user' | 'admin';
		project_assignments?: Array<{ project_id: number; role: 'viewer' | 'editor' | 'admin' }>;
		custom_temp_password?: string;
	}): Promise<{ user: AuthUser; invitation: InvitationDetails }> {
		const baseUrl = typeof window !== 'undefined' ? window.location.origin : undefined;
		return postJson('/quick_pp/uam/users/invite', {
			...payload,
			base_url: baseUrl
		});
	}

	async reinviteUser(userId: number): Promise<{ user: AuthUser; invitation: InvitationDetails }> {
		const baseUrl =
			typeof window !== 'undefined' ? encodeURIComponent(window.location.origin) : '';
		return postJson(`/quick_pp/uam/users/${userId}/reinvite?base_url=${baseUrl}`, {});
	}

	async resetUserMfa(userId: number): Promise<AuthUser> {
		const res = await postJson<{ user: AuthUser }>(`/quick_pp/uam/users/${userId}/reset-mfa`, {});
		return res.user;
	}

	async updateUser(
		userId: number,
		updates: {
			full_name?: string;
			email?: string;
			role?: 'user' | 'admin';
			is_active?: boolean;
			project_assignments?: Array<{ project_id: number; role: 'viewer' | 'editor' | 'admin' }>;
		}
	): Promise<AuthUser> {
		const res = await patchJson<{ user: AuthUser }>(`/quick_pp/uam/users/${userId}`, updates);
		if (this.user && this.user.user_id === userId) {
			this.user = res.user;
		}
		return res.user;
	}

	async deleteUser(userId: number): Promise<void> {
		await deleteJson(`/quick_pp/uam/users/${userId}`);
	}

	async listAuditLogs(limit = 50): Promise<AuditLogItem[]> {
		const res = await getJson<{ logs: AuditLogItem[] }>(`/quick_pp/uam/audit-logs?limit=${limit}`);
		return res.logs || [];
	}
}

export const authState = new AuthState();

