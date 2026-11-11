"""Model router and provider registry for agy-acpbox."""

import asyncio
from typing import Any, Dict, List, Optional, Tuple
from fastapi import HTTPException

from acpbox.config import GatewayConfig, VirtualModelConfig
from acpbox.models import ModelItem
from acpbox.providers.acp import GenericACPProvider
from acpbox.providers.base import LLMProvider
from acpbox.providers.openai_http import OpenAIHttpProvider


class ModelRouter:
    """Manages provider instances, model catalog aggregation, and 4-tier request routing."""

    def __init__(
        self,
        providers: Optional[Dict[str, LLMProvider]] = None,
        default_provider_id: str = "agy",
        virtual_models: Optional[Dict[str, VirtualModelConfig]] = None,
    ):
        self.providers: Dict[str, LLMProvider] = providers or {}
        self.default_provider_id = default_provider_id
        self.virtual_models: Dict[str, VirtualModelConfig] = virtual_models or {}

    @classmethod
    def from_config(cls, config: GatewayConfig) -> "ModelRouter":
        """Instantiates all ACP and OpenAI-compatible providers declared in GatewayConfig."""
        providers: Dict[str, LLMProvider] = {}

        # 1. Register ACP providers
        for acp_cfg in config.acp_providers:
            providers[acp_cfg.id] = GenericACPProvider(acp_cfg)

        # 2. Register OpenAI-compatible HTTP providers
        for http_cfg in config.openai_providers:
            providers[http_cfg.id] = OpenAIHttpProvider(http_cfg)

        return cls(
            providers=providers,
            default_provider_id=config.gateway.default_provider,
            virtual_models=config.virtual_models,
        )

    def register_provider(self, provider: LLMProvider) -> None:
        """Dynamically registers a provider."""
        self.providers[provider.provider_id] = provider

    async def start_all(self) -> None:
        """Concurrently starts all registered providers."""
        tasks = [p.start() for p in self.providers.values()]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def stop_all(self) -> None:
        """Concurrently stops all registered providers."""
        tasks = [p.stop() for p in self.providers.values()]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    def resolve(self, model: str) -> Tuple[LLMProvider, str]:
        """
        Resolves model name to (LLMProvider, upstream_model_id) using 4-tier logic:
        1. Virtual Model Aliases
        2. Namespaced Prefix ('provider/model' or 'provider:model')
        3. Provider Catalog Match (exact model in provider.available_models)
        4. Default Provider Fallback
        """
        if not model:
            model = "default"

        # Tier 1: Virtual Model Alias
        if model in self.virtual_models:
            vm = self.virtual_models[model]
            if vm.provider in self.providers:
                return self.providers[vm.provider], vm.upstream_model
            raise HTTPException(
                status_code=500,
                detail=f"Virtual model '{model}' targets unregistered provider '{vm.provider}'",
            )

        # Tier 2: Namespaced Prefix
        delimiter = None
        if "/" in model:
            delimiter = "/"
        elif ":" in model and not model.startswith("http"):
            delimiter = ":"

        if delimiter:
            prefix, sub_model = model.split(delimiter, 1)
            if prefix in self.providers:
                target_p = self.providers[prefix]
                avail = getattr(target_p, "available_models", [])
                # If full model string is in provider's available models (e.g. wusrouter/qwen...), keep it
                if model in avail:
                    return target_p, model
                return target_p, sub_model

        # Tier 3: Unique Provider Catalog Match
        matching_providers = []
        for p in self.providers.values():
            avail = getattr(p, "available_models", [])
            if model in avail:
                matching_providers.append(p)

        if len(matching_providers) == 1:
            return matching_providers[0], model
        elif len(matching_providers) > 1:
            # If multiple providers offer the model, prefer default provider if it is among them
            for p in matching_providers:
                if p.provider_id == self.default_provider_id:
                    return p, model
            # Otherwise return the first matching provider
            return matching_providers[0], model

        # Tier 4: Default Provider Fallback
        if self.default_provider_id in self.providers:
            return self.providers[self.default_provider_id], model

        # If default is not present but there is at least one provider, use the first
        if self.providers:
            first_p = next(iter(self.providers.values()))
            return first_p, model

        raise HTTPException(
            status_code=404,
            detail=f"No provider available to route model '{model}'",
        )

    async def list_models(self) -> List[ModelItem]:
        """Aggregates models from all providers and virtual models into a unified, deduplicated list."""
        all_models: List[ModelItem] = []
        seen_ids = set()

        # 1. Virtual models (clean aliases)
        for v_name, v_cfg in self.virtual_models.items():
            if v_name not in seen_ids:
                all_models.append(ModelItem(id=v_name, owned_by=f"virtual:{v_cfg.provider}"))
                seen_ids.add(v_name)

        # 2. Default provider models (emitted cleanly without artificial prefix)
        default_provider = self.providers.get(self.default_provider_id)
        if default_provider:
            try:
                def_models = await default_provider.get_models()
                for m in def_models:
                    if m.id not in seen_ids:
                        all_models.append(ModelItem(id=m.id, owned_by=m.owned_by))
                        seen_ids.add(m.id)
            except Exception:
                pass

        # 3. Non-default providers' models (namespaced without double-prefixing)
        for p_id, provider in self.providers.items():
            if p_id == self.default_provider_id:
                continue
            try:
                p_models = await provider.get_models()
                for m in p_models:
                    # If model is already prefixed with p_id/ or contains another namespace slash, preserve it
                    if m.id.startswith(f"{p_id}/") or "/" in m.id:
                        canonical_id = m.id
                    else:
                        canonical_id = f"{p_id}/{m.id}"

                    if canonical_id not in seen_ids:
                        all_models.append(ModelItem(id=canonical_id, owned_by=m.owned_by))
                        seen_ids.add(canonical_id)
            except Exception:
                continue

        if not all_models:
            all_models = [ModelItem(id="default", owned_by="antigravity")]

        return all_models

    async def health(self) -> Dict[str, Any]:
        """Gathers health status from all registered providers concurrently."""
        results: Dict[str, Any] = {}
        overall_healthy = False

        provider_ids = list(self.providers.keys())
        tasks = [self.providers[p_id].health() for p_id in provider_ids]
        statuses = await asyncio.gather(*tasks, return_exceptions=True)

        for p_id, res in zip(provider_ids, statuses):
            if isinstance(res, Exception):
                results[p_id] = {
                    "provider_id": p_id,
                    "healthy": False,
                    "status": "error",
                    "details": {"error": str(res)},
                }
            else:
                results[p_id] = res.model_dump()
                if res.healthy:
                    overall_healthy = True

        return {
            "status": "healthy" if overall_healthy else "unhealthy",
            "default_provider": self.default_provider_id,
            "providers_count": len(self.providers),
            "providers": results,
        }
