/// AegisCPG mobile approver -- the approval policy, as pure Dart.
///
/// Author: Pratham Kapoor <prathamkapoor027@gmail.com>
/// https://github.com/PrathamKapoor
///
/// WHAT
///     The state machine governing a step-up approval: what must be true before
///     the button enables, what a valid assertion must cover, and what happens
///     when the deadline passes.
///
/// WHY THIS FILE HAS NO WIDGETS
///     The rules that matter are properties of a transition function, not of a
///     UI. Keeping them here means they can be reasoned about -- and ported to
///     the Python implementation in `workflows/` -- without a device, a
///     simulator or a golden-file test. `main.dart` renders this state; it does
///     not decide anything.
///
/// WHY THE BINDING COVERS EVERY FIELD
///     The assertion must be bound to the session, the datum hash, both servers,
///     both lattice levels, and the challenge id. Omitting any one of them
///     creates a transplantable approval: an attacker with one legitimate
///     assertion could move it onto a different payload or a different target.
///     That is threat TV13 and it is the whole reason this type exists.
///
/// COMPLEXITY
///     O(1) per transition; `challengeBinding` is O(n) in the field lengths,
///     all of which are bounded.
library;

import 'dart:convert';

import 'package:crypto/crypto.dart';

/// The Denning lattice the control plane enforces. Must match
/// `specs/lattice_rules.smt2` and `workflows/workflows.py::LATTICE_ORDER`.
enum LatticeLevel {
  public(0, 'PUBLIC'),
  internal(1, 'INTERNAL'),
  confidential(2, 'CONFIDENTIAL'),
  secret(3, 'SECRET');

  const LatticeLevel(this.rank, this.label);

  /// Position in the lattice; higher means more sensitive.
  final int rank;

  /// The wire label, which is what the gateway and the TACR exchange.
  final String label;

  /// Parses a wire label, returning null rather than defaulting.
  ///
  /// Defaulting an unknown label to `public` would be fail-open in the one
  /// direction that matters: a release the UI believes is intra-domain when the
  /// contract treats it as a downgrade.
  static LatticeLevel? parse(String value) {
    final upper = value.toUpperCase();
    for (final level in LatticeLevel.values) {
      if (level.label == upper) {
        return level;
      }
    }
    return null;
  }
}

/// The two terminal outcomes. There is no third.
enum ApprovalOutcome {
  declassifiedAndAllowed,
  deniedAndQuarantined,
}

/// Why an approval was refused, recorded verbatim in the audit log.
///
/// Distinct reasons rather than one boolean: "denied" does not tell an incident
/// responder whether they are looking at a timeout, a forgery, or an attempt to
/// replay an approval onto a different payload.
enum DenialReason {
  notAcknowledged('not_acknowledged'),
  deadlineExpired('deadline_expired'),
  challengeMismatch('challenge_mismatch'),
  signatureInvalid('signature_invalid'),
  bindingMismatch('binding_mismatch'),
  unknownClient('unknown_client');

  const DenialReason(this.wire);

  /// The stable string stored in the audit record.
  final String wire;
}

/// One declassification the operator is being asked to authorise.
class DeclassificationRequest {
  const DeclassificationRequest({
    required this.sessionId,
    required this.datumHash,
    required this.sourceServer,
    required this.targetServer,
    required this.sourceLevel,
    required this.targetLevel,
    required this.issuedAt,
    required this.ttlSeconds,
  });

  final String sessionId;

  /// SHA-256 over the exact bytes to be released. Not the bytes: the operator
  /// cannot read them, but they can verify *which* payload they are releasing.
  final String datumHash;

  final String sourceServer;
  final String targetServer;
  final LatticeLevel sourceLevel;
  final LatticeLevel targetLevel;
  final DateTime issuedAt;
  final int ttlSeconds;

  /// Whether this release lowers the classification.
  bool get isDowngrade => targetLevel.rank < sourceLevel.rank;

  /// Whether the transition crosses a trust boundary at all.
  bool get requiresStepUp => sourceLevel != targetLevel;

  /// The exact bytes a WebAuthn assertion must cover.
  ///
  /// Every security-relevant field is present. `challengeId` is what makes one
  /// approval non-transferable to a later attempt at the same payload.
  List<int> challengeBinding(String challengeId) => utf8.encode(<String>[
        sessionId,
        datumHash,
        sourceServer,
        targetServer,
        sourceLevel.label,
        targetLevel.label,
        challengeId,
      ].join('|'));

  /// A stable digest of the binding, for display and for the audit record.
  String bindingDigest(String challengeId) =>
      sha256.convert(challengeBinding(challengeId)).toString();
}

/// The result of an approval attempt.
class ApprovalVerdict {
  const ApprovalVerdict({
    required this.outcome,
    this.reason,
    this.challengeId = '',
    this.bindingDigest = '',
  });

  final ApprovalOutcome outcome;
  final DenialReason? reason;
  final String challengeId;

  /// Present only on approval: the digest that was signed.
  final String bindingDigest;

  bool get isAllowed => outcome == ApprovalOutcome.declassifiedAndAllowed;

  Map<String, dynamic> toJson() => <String, dynamic>{
        'outcome': outcome.name,
        if (reason != null) 'reason': reason!.wire,
        if (challengeId.isNotEmpty) 'challenge_id': challengeId,
        if (bindingDigest.isNotEmpty) 'binding_digest': bindingDigest,
      };
}

/// The approval state machine.
///
/// Mirrors `workflows/workflows.py`. Both implementations exist because the
/// desktop firewall and the phone must agree on the verdict even when the phone
/// is the only device that can reach the relying party.
class ApprovalModal {
  ApprovalModal(this.request, {this.challengeId = ''});

  final DeclassificationRequest request;

  /// The outstanding challenge, if the gateway issued one.
  final String challengeId;

  bool _acknowledged = false;
  ApprovalVerdict? _verdict;

  bool get isAcknowledged => _acknowledged;
  bool get isTerminal => _verdict != null;
  ApprovalVerdict? get verdict => _verdict;

  /// Seconds before the deadline, never negative.
  int secondsRemaining(DateTime now) {
    final elapsed = now.difference(request.issuedAt).inMilliseconds / 1000.0;
    final remaining = request.ttlSeconds - elapsed;
    return remaining <= 0 ? 0 : remaining.ceil();
  }

  /// Whether the approve button should be enabled.
  bool canApprove(DateTime now) =>
      _acknowledged && _verdict == null && secondsRemaining(now) > 0;

  /// Records that the operator has read the transition. Idempotent.
  void acknowledge() => _acknowledged = true;

  /// Resolves the approval.
  ///
  /// [verifiedAssertion] is the gateway's verdict, not the device's: the phone
  /// cannot verify its own signature against a relying party it has never met.
  /// When it is null the attempt is denied for the reason given, which is why
  /// this cannot be coerced into an approval by omitting the argument.
  ApprovalVerdict resolve({
    required DateTime now,
    required String presentedChallengeId,
    required bool signatureValid,
    required bool bindingMatches,
    required String clientId,
  }) {
    final existing = _verdict;
    if (existing != null) {
      return existing;
    }

    String deny(DenialReason reason) {
      final verdict = ApprovalVerdict(
        outcome: ApprovalOutcome.deniedAndQuarantined,
        reason: reason,
        challengeId: request.sessionId,
      );
      _verdict = verdict;
      return verdict;
    }

    if (!_acknowledged) {
      return deny(DenialReason.notAcknowledged);
    }
    if (secondsRemaining(now) <= 0) {
      return deny(DenialReason.deadlineExpired);
    }
    if (presentedChallengeId != challengeId || challengeId.isEmpty) {
      return deny(DenialReason.challengeMismatch);
    }
    if (!signatureValid) {
      return deny(DenialReason.signatureInvalid);
    }
    if (!bindingMatches) {
      return deny(DenialReason.bindingMismatch);
    }
    if (clientId.isEmpty) {
      return deny(DenialReason.unknownClient);
    }

    final verdict = ApprovalVerdict(
      outcome: ApprovalOutcome.declassifiedAndAllowed,
      challengeId: challengeId,
      bindingDigest: request.bindingDigest(challengeId),
    );
    _verdict = verdict;
    return verdict;
  }
}