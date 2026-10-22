from .database_login import setup_supabase_client



def get_seed():
    
    client = setup_supabase_client("service_role")
    
    while True:
        # Find the first unused seed
        response = (
            client.table("seed")
            .select("seed")
            .eq("used", False)
            .order("seed")
            .limit(1)
            .execute()
        )

        if not response.data:
            raise RuntimeError("No unused seeds available")

        seed = response.data[0]["seed"]

        # Claim the seed, but only if it is still unused
        response = (
            client.table("seed")
            .update({"used": True})
            .eq("seed", seed)
            .eq("used", False)
            .select("seed")
            .execute()
        )

        if response.data:
            return seed

"""
def make_seeds():
    client = setup_supabase_client("service_role")

    seeds = [
        {"seed": i, "used": False}
        for i in range(1, 1001)
    ]

    response = (
        client.table("seed")
        .insert(seeds)
        .execute()
    )

    return
"""